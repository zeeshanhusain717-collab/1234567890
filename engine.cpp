// sd_engine.cpp — crash-consistent, self-verifying, sharded storage engine (on-disk format v3)
//
// Layout:
//   [0        .. 4096)     superblock copy A (used when gen is odd)
//   [4096     .. 8192)     superblock copy B (used when gen is even)
//   [8192     .. dataOff)  redo journal: begin record, entries, commit record
//   [dataOff  .. )         data slots: nonce | txn | ciphertext | tag
//
// Transaction (one write() call; serialized by txnMu_):
//   1. encrypt each touched chunk image into a journal entry bound to txn N
//   2. write begin record (fresh salt), fsync      -> invalidates any stale commit
//   3. write entries, fsync                         -> journal payload durable
//   4. write commit record (salt matches begin), fsync -> COMMIT POINT
//   5. copy entries into data slots, fsync
//   6. write superblock (gen+1, txn N, logical), fsync -> PUBLISHED
//
// Recovery (every open, and before any operation after a fault):
//   Load the newest valid superblock S. If a begin/commit pair for txn S.txn+1 exists
//   with a matching salt, verify every entry before applying any of them. All verify:
//   roll forward. Any fails: roll back, since nothing was applied. Replay is idempotent.
//
// Corruption: every slot and journal record is AEAD-authenticated and bound to its
// position and transaction. Torn writes and power loss are repaired automatically.
// Silent corruption of committed data is detected by read() and scrub() and reported.
// It cannot be repaired without a redundant copy, which this format does not carry.
//
// Threat-model limits: DRAM remanence and privileged inspectors are not defended against.
// Storage that acknowledges flushes it has not persisted voids every guarantee here, and
// software cannot detect that.
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <new>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#if defined(__ANDROID__)
#  include <android/api-level.h>
#endif

#if defined(__linux__) && (!defined(__ANDROID__) || (defined(__ANDROID_API__) && __ANDROID_API__ >= 28))
#  include <sys/random.h>
#  define SD_HAVE_GETRANDOM 1
#endif

#if defined(__linux__)
#  include <sys/prctl.h>
#endif

#if defined(__APPLE__) && defined(__has_include)
#  if __has_include(<os/proc.h>)
#    include <os/proc.h>
#    define SD_HAS_OS_PROC 1
#  endif
#endif

#define SD_API extern "C" __attribute__((visibility("default")))

static_assert(sizeof(size_t) >= 8, "64-bit address space required");
static_assert(sizeof(off_t) >= 8, "64-bit file offsets required");

typedef uint64_t sd_handle;

namespace sd {

constexpr uint64_t KiB = 1024ull;
constexpr uint64_t MiB = 1024ull * KiB;
constexpr uint64_t GiB = 1024ull * MiB;
constexpr uint64_t TiB = 1024ull * GiB;

constexpr uint64_t kMinCapacity        = 10 * MiB;
constexpr uint64_t kMaxCapacity        = 1 * TiB;
constexpr uint64_t kMinCache           = 4 * MiB;
constexpr uint64_t kMaxCache           = 512 * MiB;
constexpr uint64_t kDefaultCache       = 16 * MiB;
constexpr uint32_t kMinChunk           = 64 * 1024;
constexpr uint32_t kMaxChunk           = 16 * 1024 * 1024;
constexpr uint32_t kDefaultChunk       = 1 * 1024 * 1024;
constexpr uint64_t kMaxIoPerCall       = 32 * MiB;
constexpr uint64_t kMaxPathLen         = 4096;
constexpr uint64_t kMinFreeAfterGrowth = 1 * MiB;
constexpr uint64_t kMaxRamCacheFrac    = 8;
constexpr uint64_t kFallbackRamBytes   = 32 * MiB;

constexpr size_t   kKey          = 32;
constexpr size_t   kNonce        = 24;
constexpr size_t   kTag          = 16;
constexpr size_t   kSalt         = 16;
constexpr size_t   kAadLen       = 24;
constexpr uint32_t kMagic        = 0x53443350u;   // "SD3P"
constexpr uint32_t kVersion      = 3;
constexpr uint32_t kKindSuper    = 1;
constexpr uint32_t kKindBegin    = 2;
constexpr uint32_t kKindCommit   = 3;
constexpr uint32_t kKindChunk    = 4;
constexpr uint64_t kSbStride     = 4096;
constexpr uint64_t kMetaBytes    = 2 * kSbStride;
constexpr uint64_t kJrnBlock     = 4096;
constexpr size_t   kSbPlain      = 40;
constexpr size_t   kSbRec        = kNonce + 8 + kSbPlain + kTag;   // 88
constexpr size_t   kMetaPlain    = 40;
constexpr size_t   kMetaRec      = kNonce + kMetaPlain + kTag;     // 80
constexpr size_t   kEntryHdr     = 8;
constexpr size_t   kSlotTxnOff   = kNonce;                         // 24
constexpr size_t   kSlotCtOff    = kNonce + 8;                     // 32
constexpr size_t   kCacheShards  = 64;
constexpr size_t   kStripes      = 256;
constexpr size_t   kPoolIdle     = 8;

enum class Err : int {
    Ok = 0, NotOpen, BadConfig, BadKey, BadArg, OutOfRange,
    TooLarge, NoSpace, NoRam, Io, Crypto, Format
};

struct Config {
    uint64_t capacityBytes = 0;
    uint64_t cacheBytes    = kDefaultCache;
    uint32_t chunkBytes    = kDefaultChunk;
};

// ---------------------------------------------------------------------------
// Wiping and entropy
// ---------------------------------------------------------------------------

inline void secureWipe(void* p, size_t n) noexcept {
    if (!p || n == 0) return;
    volatile unsigned char* v = static_cast<volatile unsigned char*>(p);
    for (size_t i = 0; i < n; ++i) v[i] = 0;
#if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("" : : "r"(p) : "memory");
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

class ScopedWipe {
public:
    ScopedWipe(void* p, size_t n) noexcept : p_(p), n_(n) {}
    ~ScopedWipe() { secureWipe(p_, n_); }
    ScopedWipe(const ScopedWipe&) = delete;
    ScopedWipe& operator=(const ScopedWipe&) = delete;
private:
    void* p_;
    size_t n_;
};

inline bool constTimeEqual(const uint8_t* a, const uint8_t* b, size_t n) noexcept {
    uint8_t d = 0;
    for (size_t i = 0; i < n; ++i) d |= static_cast<uint8_t>(a[i] ^ b[i]);
    return d == 0;
}

// Kernel CSPRNG only. getrandom(2) where available; /dev/urandom otherwise.
inline bool fillRandom(uint8_t* p, size_t n) noexcept {
#if defined(SD_HAVE_GETRANDOM)
    size_t got = 0;
    bool gr = true;
    while (got < n) {
        const ssize_t r = ::getrandom(p + got, n - got, 0);
        if (r < 0) {
            if (errno == EINTR) continue;
            gr = false;
            break;
        }
        got += static_cast<size_t>(r);
    }
    if (gr) return true;
#endif
    const int fd = ::open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    size_t done = 0;
    bool ok = true;
    while (done < n) {
        const ssize_t r = ::read(fd, p + done, n - done);
        if (r < 0) {
            if (errno == EINTR) continue;
            ok = false;
            break;
        }
        if (r == 0) { ok = false; break; }
        done += static_cast<size_t>(r);
    }
    ::close(fd);
    return ok;
}

inline size_t pageSize() noexcept {
    static const size_t p = [] {
        const long v = ::sysconf(_SC_PAGESIZE);
        return v > 0 ? static_cast<size_t>(v) : size_t{4096};
    }();
    return p;
}

// Opt-in, process-wide.
inline void hardenProcess() noexcept {
#if defined(__linux__)
    ::prctl(PR_SET_DUMPABLE, 0, 0, 0, 0);
#endif
    struct rlimit rl{};
    if (::getrlimit(RLIMIT_CORE, &rl) == 0) {
        rl.rlim_cur = 0;
        ::setrlimit(RLIMIT_CORE, &rl);
    }
}

inline uint64_t availableRamBytes() noexcept {
#if defined(SD_HAS_OS_PROC)
    return static_cast<uint64_t>(os_proc_available_memory());
#elif defined(__linux__)
    std::FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    uint64_t out = 0;
    while (std::fgets(line, sizeof line, f)) {
        unsigned long long kb = 0;
        if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
            out = static_cast<uint64_t>(kb) * KiB;
            break;
        }
    }
    std::fclose(f);
    return out;
#else
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// Secure memory
// ---------------------------------------------------------------------------

class SecureBytes {
public:
    SecureBytes() noexcept = default;
    explicit SecureBytes(size_t n) { allocate(n); }
    ~SecureBytes() { release(); }

    SecureBytes(const SecureBytes&) = delete;
    SecureBytes& operator=(const SecureBytes&) = delete;

    SecureBytes(SecureBytes&& o) noexcept { take(o); }
    SecureBytes& operator=(SecureBytes&& o) noexcept {
        if (this != &o) {
            release();
            take(o);
        }
        return *this;
    }

    uint8_t* data() noexcept { return usable_; }
    const uint8_t* data() const noexcept { return usable_; }
    size_t size() const noexcept { return len_; }
    bool pinned() const noexcept { return locked_; }

    void release() noexcept {
        if (!base_) return;
        secureWipe(usable_, span_);
        if (locked_) ::munlock(usable_, span_);
        ::munmap(base_, mapLen_);
        clearFields();
    }

private:
    void allocate(size_t n) {
        if (n == 0) return;
        const size_t page = pageSize();
        if (n > std::numeric_limits<size_t>::max() - 3 * page) throw std::bad_alloc();
        const size_t span = (n + page - 1) / page * page;
        const size_t total = span + 2 * page;

        void* m = ::mmap(nullptr, total, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED) throw std::bad_alloc();

        uint8_t* base = static_cast<uint8_t*>(m);
        if (::mprotect(base, page, PROT_NONE) != 0 ||
            ::mprotect(base + page + span, page, PROT_NONE) != 0) {
            ::munmap(base, total);
            throw std::bad_alloc();
        }

        base_ = base;
        mapLen_ = total;
        usable_ = base + page;
        span_ = span;
        len_ = n;

#if defined(MADV_DONTDUMP)
        ::madvise(usable_, span_, MADV_DONTDUMP);
#endif
#if defined(MADV_DONTFORK)
        ::madvise(usable_, span_, MADV_DONTFORK);
#endif
#if defined(MADV_WIPEONFORK)
        ::madvise(usable_, span_, MADV_WIPEONFORK);
#endif
        locked_ = (::mlock(usable_, span_) == 0);
    }

    void take(SecureBytes& o) noexcept {
        base_ = o.base_;
        mapLen_ = o.mapLen_;
        usable_ = o.usable_;
        span_ = o.span_;
        len_ = o.len_;
        locked_ = o.locked_;
        o.clearFields();
    }

    void clearFields() noexcept {
        base_ = nullptr;
        usable_ = nullptr;
        mapLen_ = 0;
        span_ = 0;
        len_ = 0;
        locked_ = false;
    }

    uint8_t* base_ = nullptr;
    uint8_t* usable_ = nullptr;
    size_t mapLen_ = 0;
    size_t span_ = 0;
    size_t len_ = 0;
    bool locked_ = false;
};

// Recycles zeroized secure buffers.
class BufferPool : public std::enable_shared_from_this<BufferPool> {
public:
    BufferPool(size_t bufBytes, size_t maxIdle) : bufBytes_(bufBytes), maxIdle_(maxIdle) {}

    std::shared_ptr<SecureBytes> acquire() {
        std::unique_ptr<SecureBytes> b;
        {
            std::lock_guard<std::mutex> g(mu_);
            if (!idle_.empty()) {
                b = std::move(idle_.back());
                idle_.pop_back();
            }
        }
        if (!b) b = std::make_unique<SecureBytes>(bufBytes_);
        std::shared_ptr<BufferPool> self = shared_from_this();
        SecureBytes* raw = b.release();
        return std::shared_ptr<SecureBytes>(raw, [self](SecureBytes* p) { self->recycle(p); });
    }

private:
    void recycle(SecureBytes* p) noexcept {
        std::unique_ptr<SecureBytes> u(p);
        secureWipe(p->data(), p->size());
        try {
            std::lock_guard<std::mutex> g(mu_);
            if (idle_.size() < maxIdle_) idle_.push_back(std::move(u));
        } catch (...) {}
    }

    size_t bufBytes_;
    size_t maxIdle_;
    std::mutex mu_;
    std::vector<std::unique_ptr<SecureBytes>> idle_;
};

// ---------------------------------------------------------------------------
// Crypto primitives (RFC 8439 ChaCha20/Poly1305; XChaCha20 construction)
// ---------------------------------------------------------------------------

namespace crypto {
namespace detail {
inline uint32_t rotl(uint32_t x, int n) noexcept { return (x << n) | (x >> (32 - n)); }
inline uint32_t ld32(const uint8_t* p) noexcept {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
inline void st32(uint8_t* p, uint32_t v) noexcept {
    p[0] = uint8_t(v); p[1] = uint8_t(v >> 8); p[2] = uint8_t(v >> 16); p[3] = uint8_t(v >> 24);
}
inline void quarter(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) noexcept {
    a += b; d ^= a; d = rotl(d, 16);
    c += d; b ^= c; b = rotl(b, 12);
    a += b; d ^= a; d = rotl(d, 8);
    c += d; b ^= c; b = rotl(b, 7);
}
inline void rounds20(uint32_t x[16]) noexcept {
    for (int i = 0; i < 10; ++i) {
        quarter(x[0], x[4], x[8],  x[12]);
        quarter(x[1], x[5], x[9],  x[13]);
        quarter(x[2], x[6], x[10], x[14]);
        quarter(x[3], x[7], x[11], x[15]);
        quarter(x[0], x[5], x[10], x[15]);
        quarter(x[1], x[6], x[11], x[12]);
        quarter(x[2], x[7], x[8],  x[13]);
        quarter(x[3], x[4], x[9],  x[14]);
    }
}
inline void initState(uint32_t x[16], const uint8_t key[32]) noexcept {
    x[0] = 0x61707865; x[1] = 0x3320646e; x[2] = 0x79622d32; x[3] = 0x6b206574;
    for (int i = 0; i < 8; ++i) x[4 + i] = ld32(key + 4 * i);
}
constexpr uint32_t M26 = 0x3ffffff;
} // namespace detail

inline void hchacha20(const uint8_t key[32], const uint8_t nonce16[16], uint8_t out[32]) noexcept {
    uint32_t x[16];
    ScopedWipe wx(x, sizeof x);
    detail::initState(x, key);
    for (int i = 0; i < 4; ++i) x[12 + i] = detail::ld32(nonce16 + 4 * i);
    detail::rounds20(x);
    const int idx[8] = {0, 1, 2, 3, 12, 13, 14, 15};
    for (int i = 0; i < 8; ++i) detail::st32(out + 4 * i, x[idx[i]]);
}

inline void chacha20Xor(const uint8_t key[32], const uint8_t nonce[12], uint32_t counter,
                        uint8_t* data, size_t len) noexcept {
    uint32_t init[16];
    uint32_t x[16];
    uint8_t ks[64];
    ScopedWipe wInit(init, sizeof init);
    ScopedWipe wX(x, sizeof x);
    ScopedWipe wKs(ks, sizeof ks);

    detail::initState(init, key);
    init[12] = counter;
    for (int i = 0; i < 3; ++i) init[13 + i] = detail::ld32(nonce + 4 * i);
    while (len > 0) {
        for (int i = 0; i < 16; ++i) x[i] = init[i];
        detail::rounds20(x);
        for (int i = 0; i < 16; ++i) detail::st32(ks + 4 * i, x[i] + init[i]);
        const size_t n = len < 64 ? len : 64;
        for (size_t i = 0; i < n; ++i) data[i] ^= ks[i];
        data += n;
        len -= n;
        ++init[12];
    }
}

class Poly1305 {
public:
    explicit Poly1305(const uint8_t key[32]) noexcept {
        using namespace detail;
        r_[0] = ld32(key + 0) & 0x3ffffff;
        r_[1] = (ld32(key + 3) >> 2) & 0x3ffff03;
        r_[2] = (ld32(key + 6) >> 4) & 0x3ffc0ff;
        r_[3] = (ld32(key + 9) >> 6) & 0x3f03fff;
        r_[4] = (ld32(key + 12) >> 8) & 0x00fffff;
        for (int i = 0; i < 4; ++i) pad_[i] = ld32(key + 16 + 4 * i);
    }
    ~Poly1305() { secureWipe(this, sizeof *this); }
    Poly1305(const Poly1305&) = delete;
    Poly1305& operator=(const Poly1305&) = delete;

    void update(const uint8_t* m, size_t n) noexcept {
        if (leftover_) {
            size_t want = 16 - leftover_;
            if (want > n) want = n;
            std::memcpy(buf_ + leftover_, m, want);
            n -= want; m += want; leftover_ += want;
            if (leftover_ < 16) return;
            blocks(buf_, 16, false);
            leftover_ = 0;
        }
        if (n >= 16) {
            const size_t full = n & ~size_t(15);
            blocks(m, full, false);
            m += full; n -= full;
        }
        if (n) { std::memcpy(buf_, m, n); leftover_ = n; }
    }

    void finish(uint8_t mac[16]) noexcept {
        using namespace detail;
        if (leftover_) {
            buf_[leftover_] = 1;
            for (size_t i = leftover_ + 1; i < 16; ++i) buf_[i] = 0;
            blocks(buf_, 16, true);
            leftover_ = 0;
        }
        uint32_t h0 = h_[0], h1 = h_[1], h2 = h_[2], h3 = h_[3], h4 = h_[4];
        uint32_t c = h1 >> 26; h1 &= M26;
        h2 += c; c = h2 >> 26; h2 &= M26;
        h3 += c; c = h3 >> 26; h3 &= M26;
        h4 += c; c = h4 >> 26; h4 &= M26;
        h0 += c * 5; c = h0 >> 26; h0 &= M26;
        h1 += c;

        uint32_t g0 = h0 + 5; c = g0 >> 26; g0 &= M26;
        uint32_t g1 = h1 + c; c = g1 >> 26; g1 &= M26;
        uint32_t g2 = h2 + c; c = g2 >> 26; g2 &= M26;
        uint32_t g3 = h3 + c; c = g3 >> 26; g3 &= M26;
        uint32_t g4 = h4 + c - (1u << 26);

        uint32_t mask = (g4 >> 31) - 1;
        g0 &= mask; g1 &= mask; g2 &= mask; g3 &= mask; g4 &= mask;
        mask = ~mask;
        h0 = (h0 & mask) | g0; h1 = (h1 & mask) | g1; h2 = (h2 & mask) | g2;
        h3 = (h3 & mask) | g3; h4 = (h4 & mask) | g4;

        h0 = h0 | (h1 << 26);
        h1 = (h1 >> 6) | (h2 << 20);
        h2 = (h2 >> 12) | (h3 << 14);
        h3 = (h3 >> 18) | (h4 << 8);

        uint64_t f = uint64_t(h0) + pad_[0]; st32(mac + 0, uint32_t(f));
        f = uint64_t(h1) + pad_[1] + (f >> 32); st32(mac + 4, uint32_t(f));
        f = uint64_t(h2) + pad_[2] + (f >> 32); st32(mac + 8, uint32_t(f));
        f = uint64_t(h3) + pad_[3] + (f >> 32); st32(mac + 12, uint32_t(f));

        secureWipe(h_, sizeof h_);
        secureWipe(r_, sizeof r_);
        secureWipe(pad_, sizeof pad_);
        secureWipe(buf_, sizeof buf_);
    }

private:
    void blocks(const uint8_t* m, size_t bytes, bool fin) noexcept {
        using namespace detail;
        const uint32_t hibit = fin ? 0u : (1u << 24);
        const uint32_t r0 = r_[0], r1 = r_[1], r2 = r_[2], r3 = r_[3], r4 = r_[4];
        const uint32_t s1 = r1 * 5, s2 = r2 * 5, s3 = r3 * 5, s4 = r4 * 5;
        uint32_t h0 = h_[0], h1 = h_[1], h2 = h_[2], h3 = h_[3], h4 = h_[4];

        while (bytes >= 16) {
            h0 += ld32(m + 0) & M26;
            h1 += (ld32(m + 3) >> 2) & M26;
            h2 += (ld32(m + 6) >> 4) & M26;
            h3 += (ld32(m + 9) >> 6) & M26;
            h4 += (ld32(m + 12) >> 8) | hibit;

            uint64_t d0 = uint64_t(h0)*r0 + uint64_t(h1)*s4 + uint64_t(h2)*s3 + uint64_t(h3)*s2 + uint64_t(h4)*s1;
            uint64_t d1 = uint64_t(h0)*r1 + uint64_t(h1)*r0 + uint64_t(h2)*s4 + uint64_t(h3)*s3 + uint64_t(h4)*s2;
            uint64_t d2 = uint64_t(h0)*r2 + uint64_t(h1)*r1 + uint64_t(h2)*r0 + uint64_t(h3)*s4 + uint64_t(h4)*s3;
            uint64_t d3 = uint64_t(h0)*r3 + uint64_t(h1)*r2 + uint64_t(h2)*r1 + uint64_t(h3)*r0 + uint64_t(h4)*s4;
            uint64_t d4 = uint64_t(h0)*r4 + uint64_t(h1)*r3 + uint64_t(h2)*r2 + uint64_t(h3)*r1 + uint64_t(h4)*r0;

            uint32_t c = uint32_t(d0 >> 26); h0 = uint32_t(d0) & M26;
            d1 += c; c = uint32_t(d1 >> 26); h1 = uint32_t(d1) & M26;
            d2 += c; c = uint32_t(d2 >> 26); h2 = uint32_t(d2) & M26;
            d3 += c; c = uint32_t(d3 >> 26); h3 = uint32_t(d3) & M26;
            d4 += c; c = uint32_t(d4 >> 26); h4 = uint32_t(d4) & M26;
            h0 += c * 5; c = h0 >> 26; h0 &= M26;
            h1 += c;

            m += 16;
            bytes -= 16;
        }
        h_[0] = h0; h_[1] = h1; h_[2] = h2; h_[3] = h3; h_[4] = h4;
    }

    uint32_t r_[5]{}, h_[5]{}, pad_[4]{};
    uint8_t buf_[16]{};
    size_t leftover_ = 0;
};

inline void aeadTag(const uint8_t key[32], const uint8_t nonce[12], const uint8_t* aad, size_t aadLen,
                    const uint8_t* ct, size_t ctLen, uint8_t tag[16]) noexcept {
    uint8_t polyKey[32] = {0};
    ScopedWipe wPk(polyKey, sizeof polyKey);
    chacha20Xor(key, nonce, 0, polyKey, sizeof polyKey);

    Poly1305 mac(polyKey);
    static const uint8_t zeros[16] = {0};
    uint8_t lens[16];
    for (int i = 0; i < 8; ++i) {
        lens[i]     = uint8_t(uint64_t(aadLen) >> (8 * i));
        lens[8 + i] = uint8_t(uint64_t(ctLen) >> (8 * i));
    }
    mac.update(aad, aadLen);
    mac.update(zeros, (16 - aadLen % 16) % 16);
    mac.update(ct, ctLen);
    mac.update(zeros, (16 - ctLen % 16) % 16);
    mac.update(lens, 16);
    mac.finish(tag);
}

inline void xSeal(const uint8_t key[32], const uint8_t nonce24[24],
                  const uint8_t* aad, size_t aadLen,
                  const uint8_t* pt, size_t len, uint8_t* ct, uint8_t tag[16]) noexcept {
    uint8_t sub[32];
    ScopedWipe wSub(sub, sizeof sub);
    uint8_t n12[12] = {0, 0, 0, 0};
    hchacha20(key, nonce24, sub);
    std::memcpy(n12 + 4, nonce24 + 16, 8);
    if (ct != pt && len) std::memcpy(ct, pt, len);
    chacha20Xor(sub, n12, 1, ct, len);
    aeadTag(sub, n12, aad, aadLen, ct, len, tag);
}

inline bool xOpen(const uint8_t key[32], const uint8_t nonce24[24],
                  const uint8_t* aad, size_t aadLen,
                  const uint8_t* ct, size_t len, const uint8_t tag[16], uint8_t* pt) noexcept {
    uint8_t sub[32];
    uint8_t expect[16];
    ScopedWipe wSub(sub, sizeof sub);
    ScopedWipe wExp(expect, sizeof expect);
    uint8_t n12[12] = {0, 0, 0, 0};
    hchacha20(key, nonce24, sub);
    std::memcpy(n12 + 4, nonce24 + 16, 8);
    aeadTag(sub, n12, aad, aadLen, ct, len, expect);
    if (!constTimeEqual(expect, tag, 16)) return false;
    if (pt != ct && len) std::memcpy(pt, ct, len);
    chacha20Xor(sub, n12, 1, pt, len);
    return true;
}

// RFC 8439 Section 2.5.2 test vector. Run on every open; fails closed.
inline bool poly1305SelfTest() noexcept {
    const uint8_t key[32] = {
        0x85,0xd6,0xbe,0x78,0x57,0x55,0x6d,0x33,0x7f,0x44,0x52,0xfe,0x42,0xd5,0x06,0xa8,
        0x01,0x03,0x80,0x8a,0xfb,0x0d,0xb2,0xfd,0x4a,0xbf,0xf6,0xaf,0x41,0x49,0xf5,0x1b};
    const uint8_t expect[16] = {
        0xa8,0x06,0x1d,0xc1,0x30,0x51,0x36,0xc6,0xc2,0x2b,0x8b,0xaf,0x0c,0x01,0x27,0xa9};
    const char* msg = "Cryptographic Forum Research Group";
    Poly1305 p(key);
    p.update(reinterpret_cast<const uint8_t*>(msg), std::strlen(msg));
    uint8_t tag[16];
    p.finish(tag);
    return constTimeEqual(tag, expect, 16);
}
} // namespace crypto

// ---------------------------------------------------------------------------
// Storage backends. Only ciphertext crosses this boundary.
// ---------------------------------------------------------------------------

class IBlockStore {
public:
    virtual ~IBlockStore() = default;
    virtual bool read(uint64_t off, uint8_t* buf, size_t n) = 0;
    virtual bool write(uint64_t off, const uint8_t* buf, size_t n) = 0;
    virtual bool sync() = 0;
    virtual bool size(uint64_t& out) const = 0;
    virtual bool freeSpace(uint64_t& out) const = 0;
    virtual bool erase() = 0;
    virtual void close() noexcept = 0;
};

class FileBlockStore final : public IBlockStore {
public:
    static std::unique_ptr<FileBlockStore> create(const std::string& dir) {
        if (dir.empty() || dir.size() > kMaxPathLen) return nullptr;
        std::unique_ptr<FileBlockStore> s(new FileBlockStore(dir));
        if (!s->openFile()) return nullptr;
        return s;
    }

    ~FileBlockStore() override { close(); }

    bool read(uint64_t off, uint8_t* b, size_t n) override { return ioRead(off, b, n); }
    bool write(uint64_t off, const uint8_t* b, size_t n) override { return ioWrite(off, b, n); }

    bool sync() override { return fd_ >= 0 && ::fsync(fd_) == 0; }

    bool size(uint64_t& out) const override {
        struct stat st{};
        if (fd_ < 0 || ::fstat(fd_, &st) != 0 || st.st_size < 0) return false;
        out = static_cast<uint64_t>(st.st_size);
        return true;
    }

    bool freeSpace(uint64_t& out) const override {
        struct statvfs sv{};
        if (::statvfs(dir_.c_str(), &sv) != 0) return false;
        out = static_cast<uint64_t>(sv.f_bavail) * static_cast<uint64_t>(sv.f_frsize);
        return true;
    }

    // Overwrites ciphertext with CSPRNG output, fsyncs, then unlinks. Best effort only:
    // flash translation layers may retain old physical pages.
    bool erase() override {
        if (fd_ >= 0) {
            uint64_t sz = 0;
            if (size(sz) && sz > 0) scrub(sz);
            close();
        }
        if (::unlink(path_.c_str()) == 0) {
            fsyncDir();
            return true;
        }
        return errno == ENOENT;
    }

    void close() noexcept override {
        if (fd_ >= 0) {
            ::close(fd_);   // also releases the flock
            fd_ = -1;
        }
    }

private:
    explicit FileBlockStore(std::string dir) : dir_(std::move(dir)), path_(dir_ + "/sd_store.bin") {}

    static bool inRange(uint64_t off, size_t n) noexcept {
        const uint64_t lim = static_cast<uint64_t>(std::numeric_limits<off_t>::max());
        return off <= lim && static_cast<uint64_t>(n) <= lim - off;
    }

    bool openFile() {
        fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (fd_ < 0) return false;
        struct stat st{};
        if (::fstat(fd_, &st) != 0 || !S_ISREG(st.st_mode)) {
            close();
            return false;
        }
        if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
            close();
            return false;
        }
        if (::fchmod(fd_, 0600) != 0) {
            close();
            return false;
        }
        fsyncDir();
        return true;
    }

    void fsyncDir() noexcept {
        const int dfd = ::open(dir_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (dfd >= 0) {
            ::fsync(dfd);
            ::close(dfd);
        }
    }

    bool ioRead(uint64_t off, uint8_t* p, size_t n) const noexcept {
        if (fd_ < 0 || !inRange(off, n)) return false;
        size_t done = 0;
        while (done < n) {
            const ssize_t r = ::pread(fd_, p + done, n - done, static_cast<off_t>(off + done));
            if (r < 0) {
                if (errno == EINTR) continue;
                return false;
            }
            if (r == 0) return false;
            done += static_cast<size_t>(r);
        }
        return true;
    }

    bool ioWrite(uint64_t off, const uint8_t* p, size_t n) noexcept {
        if (fd_ < 0 || !inRange(off, n)) return false;
        size_t done = 0;
        while (done < n) {
            const ssize_t r = ::pwrite(fd_, p + done, n - done, static_cast<off_t>(off + done));
            if (r < 0) {
                if (errno == EINTR) continue;
                return false;
            }
            if (r == 0) return false;
            done += static_cast<size_t>(r);
        }
        return true;
    }

    bool scrub(uint64_t total) noexcept {
        std::vector<uint8_t> buf(64 * 1024);
        ScopedWipe wBuf(buf.data(), buf.size());
        uint64_t off = 0;
        while (off < total) {
            const size_t n = static_cast<size_t>(std::min<uint64_t>(buf.size(), total - off));
            if (!fillRandom(buf.data(), n)) return false;
            if (!ioWrite(off, buf.data(), n)) return false;
            off += n;
        }
        return ::fsync(fd_) == 0;
    }

    std::string dir_;
    std::string path_;
    int fd_ = -1;
};

// RAM-backed store for tests and tooling only.
class MemoryBlockStore final : public IBlockStore {
public:
    explicit MemoryBlockStore(uint64_t quota = 0)
        : quota_(quota ? quota : std::numeric_limits<uint64_t>::max()) {}
    ~MemoryBlockStore() override { erase(); }

    bool read(uint64_t off, uint8_t* b, size_t n) override {
        if (off > data_.size() || n > data_.size() - off) return false;
        std::memcpy(b, data_.data() + off, n);
        return true;
    }

    bool write(uint64_t off, const uint8_t* b, size_t n) override {
        if (off > quota_ || static_cast<uint64_t>(n) > quota_ - off) return false;
        const uint64_t need = off + static_cast<uint64_t>(n);
        if (need > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) return false;
        if (need > data_.size()) data_.resize(static_cast<size_t>(need), 0);
        std::memcpy(data_.data() + off, b, n);
        return true;
    }

    bool sync() override { return true; }
    bool size(uint64_t& out) const override { out = data_.size(); return true; }
    bool freeSpace(uint64_t& out) const override {
        const uint64_t used = data_.size();
        out = quota_ > used ? quota_ - used : 0;
        return true;
    }
    bool erase() override {
        if (!data_.empty()) secureWipe(data_.data(), data_.size());
        data_.clear();
        return true;
    }
    void close() noexcept override {}

private:
    std::vector<uint8_t> data_;
    uint64_t quota_;
};

// ---------------------------------------------------------------------------
// Format helpers
// ---------------------------------------------------------------------------

namespace detail {
inline void put32(uint8_t* p, uint32_t v) noexcept { for (int i = 0; i < 4; ++i) p[i] = uint8_t(v >> (8 * i)); }
inline void put64(uint8_t* p, uint64_t v) noexcept { for (int i = 0; i < 8; ++i) p[i] = uint8_t(v >> (8 * i)); }
inline uint32_t get32(const uint8_t* p) noexcept {
    uint32_t v = 0; for (int i = 0; i < 4; ++i) v |= uint32_t(p[i]) << (8 * i); return v;
}
inline uint64_t get64(const uint8_t* p) noexcept {
    uint64_t v = 0; for (int i = 0; i < 8; ++i) v |= uint64_t(p[i]) << (8 * i); return v;
}
// AAD binds every record to its position (idx), chunk size, kind, and transaction.
inline void makeAad(uint8_t (&a)[kAadLen], uint64_t idx, uint32_t cb, uint32_t kind, uint64_t txn) noexcept {
    put64(a, idx); put32(a + 8, cb); put32(a + 12, kind); put64(a + 16, txn);
}
inline uint64_t ceilDiv(uint64_t a, uint64_t b) noexcept { return a / b + (a % b != 0 ? 1 : 0); }
inline uint64_t roundUp(uint64_t v, uint64_t m) noexcept { return ceilDiv(v, m) * m; }
inline bool isPow2(uint32_t v) noexcept { return v != 0 && (v & (v - 1)) == 0; }
} // namespace detail

inline uint64_t superOff(uint64_t gen) noexcept { return (gen & 1) ? kSbStride : 0; }

inline uint64_t clampCache(uint64_t req) noexcept {
    const uint64_t ram = availableRamBytes();
    uint64_t hi = kMaxCache;
    hi = (std::min)(hi, ram ? ram / kMaxRamCacheFrac : kFallbackRamBytes);
    return (std::max)((std::min)(req, hi), kMinCache);
}

// ---------------------------------------------------------------------------
// Chunk-stripe range lock. Acquired in ascending order, which prevents deadlock.
// ---------------------------------------------------------------------------

using StripeArray = std::array<std::shared_mutex, kStripes>;

class StripeSet {
public:
    StripeSet(StripeArray& s, uint64_t firstChunk, uint64_t lastChunk, bool exclusive)
        : s_(s), excl_(exclusive) {
        if (lastChunk - firstChunk + 1 >= kStripes) {
            ids_.reserve(kStripes);
            for (uint32_t i = 0; i < kStripes; ++i) ids_.push_back(i);
        } else {
            for (uint64_t c = firstChunk; c <= lastChunk; ++c) {
                ids_.push_back(static_cast<uint32_t>(c % kStripes));
            }
            std::sort(ids_.begin(), ids_.end());
            ids_.erase(std::unique(ids_.begin(), ids_.end()), ids_.end());
        }
        size_t n = 0;
        try {
            for (; n < ids_.size(); ++n) lockOne(ids_[n]);
        } catch (...) {
            while (n > 0) unlockOne(ids_[--n]);
            throw;
        }
    }

    ~StripeSet() {
        for (auto it = ids_.rbegin(); it != ids_.rend(); ++it) unlockOne(*it);
    }

    StripeSet(const StripeSet&) = delete;
    StripeSet& operator=(const StripeSet&) = delete;

private:
    void lockOne(uint32_t id) { excl_ ? s_[id].lock() : s_[id].lock_shared(); }
    void unlockOne(uint32_t id) noexcept { excl_ ? s_[id].unlock() : s_[id].unlock_shared(); }

    StripeArray& s_;
    bool excl_;
    std::vector<uint32_t> ids_;
};

// ---------------------------------------------------------------------------
// Persistent records
// ---------------------------------------------------------------------------

struct Super {
    uint64_t gen = 0;
    uint64_t txn = 0;
    uint64_t logical = 0;
    uint64_t capacity = 0;
    uint32_t chunk = 0;
};

struct MetaRec {
    uint64_t txn = 0;
    uint64_t count = 0;
    uint64_t logical = 0;
    uint8_t salt[kSalt] = {};
};

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

class Engine {
public:
    struct Health {
        uint64_t committedTxn = 0;
        uint64_t rolledForward = 0;
        uint64_t rolledBack = 0;
        bool faulted = false;
    };

    static Err open(const Config& cfg, std::unique_ptr<IBlockStore> store,
                    const uint8_t* key, size_t keyLen, std::unique_ptr<Engine>& out) noexcept;

    ~Engine() { close(); }
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    Err write(uint64_t off, const uint8_t* data, uint64_t len) noexcept;

    // Input-only view. Plaintext is never returned as a view.
    Err write(uint64_t off, std::string_view data) noexcept {
        return write(off, reinterpret_cast<const uint8_t*>(data.data()), static_cast<uint64_t>(data.size()));
    }

    Err read(uint64_t off, uint8_t* out, uint64_t len) noexcept;
    Err resize(uint64_t newCapacity) noexcept;
    Err destroy() noexcept;
    void close() noexcept;
    Err scrub(uint64_t& badChunks) noexcept;
    Err health(Health& h) const noexcept;
    void setCacheBytes(uint64_t requested) noexcept;

    uint64_t size() const noexcept { return logical_.load(std::memory_order_acquire); }
    uint64_t capacity() const noexcept { return capacity_.load(std::memory_order_acquire); }

private:
    using Chunk = std::shared_ptr<SecureBytes>;
    using LruList = std::list<std::pair<uint64_t, Chunk>>;

    struct alignas(64) CacheShard {
        std::mutex mu;
        LruList lru;
        std::unordered_map<uint64_t, LruList::iterator> index;
        size_t maxEntries = 0;
    };

    Engine() = default;

    void setLayout(uint32_t cb) noexcept;
    uint64_t slotOffset(uint64_t idx) const noexcept { return dataOff_ + idx * slotBytes_; }

    bool decodeSuper(const uint8_t* rec, Super& out) const;
    bool loadBestSuper(Super& best);
    Err writeSuperLocked(const Super& s);
    bool encodeMeta(uint32_t kind, const MetaRec& m, uint8_t* out) const;
    bool decodeMeta(uint32_t kind, const uint8_t* in, MetaRec& m) const;

    Err recoverLocked() noexcept;
    Err verifyEntries(const uint8_t* entries, uint64_t n, const MetaRec& m) noexcept;
    Err applyCommittedLocked(const uint8_t* entries, uint64_t n, const MetaRec& m) noexcept;
    Err healIfFaulted() noexcept;
    Err loadChunk(uint64_t idx, Chunk& out, bool cache) noexcept;

    Chunk cacheGet(uint64_t idx) noexcept;
    void cachePut(uint64_t idx, Chunk p) noexcept;
    void cacheErase(uint64_t idx) noexcept;
    static void eraseIndexLocked(CacheShard& s, uint64_t idx) noexcept;
    static void evictShardLocked(CacheShard& s) noexcept;
    void applyCacheLocked();
    void trimCacheLocked() noexcept;

    std::unique_ptr<IBlockStore> store_;
    SecureBytes key_;
    std::shared_ptr<BufferPool> pool_;
    Super sb_;                                       // guarded by txnMu_ or exclusive mu_
    bool open_ = false;                              // guarded by mu_
    uint32_t chunkBytes_ = 0;                        // immutable after open
    uint64_t slotBytes_ = 0;
    uint64_t entryBytes_ = 0;
    uint64_t maxEntries_ = 0;
    uint64_t dataOff_ = 0;

    std::atomic<uint64_t> capacity_{0};
    std::atomic<uint64_t> logical_{0};
    std::atomic<uint64_t> committedTxn_{0};
    std::atomic<uint64_t> rolledForward_{0};
    std::atomic<uint64_t> rolledBack_{0};
    std::atomic<bool> faulted_{false};

    mutable std::shared_mutex mu_;                   // session: data ops shared, lifecycle exclusive
    StripeArray stripes_;                            // chunk-range locks
    std::mutex txnMu_;                               // serializes write transactions
    std::mutex cfgMu_;
    uint64_t requestedCache_ = 0;
    std::array<CacheShard, kCacheShards> shards_;
};

// Journal and data layout derived from chunk size. The journal holds the largest possible
// write (kMaxIoPerCall) plus one boundary chunk on each side.
inline void Engine::setLayout(uint32_t cb) noexcept {
    chunkBytes_ = cb;
    slotBytes_ = kNonce + 8 + static_cast<uint64_t>(cb) + kTag;
    entryBytes_ = kEntryHdr + slotBytes_;
    maxEntries_ = kMaxIoPerCall / cb + 2;
    const uint64_t journalBytes = detail::roundUp(kJrnBlock + maxEntries_ * entryBytes_ + kMetaRec, kJrnBlock);
    dataOff_ = kMetaBytes + journalBytes;
}

inline bool Engine::decodeSuper(const uint8_t* rec, Super& out) const {
    const uint64_t gen = detail::get64(rec + 24);
    if (gen == 0) return false;
    uint8_t aad[kAadLen];
    detail::makeAad(aad, gen, 0, kKindSuper, 0);
    uint8_t pt[kSbPlain];
    ScopedWipe wPt(pt, sizeof pt);
    if (!crypto::xOpen(key_.data(), rec, aad, kAadLen, rec + 32, kSbPlain, rec + 32 + kSbPlain, pt)) {
        return false;
    }
    if (detail::get32(pt) != kMagic || detail::get32(pt + 4) != kVersion) return false;
    Super s;
    s.gen = gen;
    s.chunk = detail::get32(pt + 8);
    s.capacity = detail::get64(pt + 16);
    s.logical = detail::get64(pt + 24);
    s.txn = detail::get64(pt + 32);
    if (s.capacity < kMinCapacity || s.capacity > kMaxCapacity || s.logical > s.capacity) return false;
    out = s;
    return true;
}

// Reads both superblock copies and returns the newest one that authenticates. A copy is
// only accepted from the offset its generation parity dictates.
inline bool Engine::loadBestSuper(Super& best) {
    bool found = false;
    for (uint64_t slot = 0; slot < 2; ++slot) {
        uint8_t rec[kSbRec];
        if (!store_->read(slot * kSbStride, rec, kSbRec)) continue;
        Super s;
        if (!decodeSuper(rec, s)) continue;
        if (((s.gen & 1) != 0) != (slot == 1)) continue;
        if (!found || s.gen > best.gen) {
            best = s;
            found = true;
        }
    }
    return found;
}

inline Err Engine::writeSuperLocked(const Super& s) {
    uint8_t pt[kSbPlain] = {0};
    ScopedWipe wPt(pt, sizeof pt);
    detail::put32(pt, kMagic);
    detail::put32(pt + 4, kVersion);
    detail::put32(pt + 8, s.chunk);
    detail::put64(pt + 16, s.capacity);
    detail::put64(pt + 24, s.logical);
    detail::put64(pt + 32, s.txn);

    uint8_t rec[kSbRec];
    if (!fillRandom(rec, kNonce)) return Err::Crypto;
    detail::put64(rec + 24, s.gen);
    uint8_t aad[kAadLen];
    detail::makeAad(aad, s.gen, 0, kKindSuper, 0);
    crypto::xSeal(key_.data(), rec, aad, kAadLen, pt, kSbPlain, rec + 32, rec + 32 + kSbPlain);
    if (!store_->write(superOff(s.gen), rec, kSbRec)) return Err::Io;
    if (!store_->sync()) return Err::Io;
    return Err::Ok;
}

inline bool Engine::encodeMeta(uint32_t kind, const MetaRec& m, uint8_t* out) const {
    uint8_t pt[kMetaPlain] = {0};
    ScopedWipe wPt(pt, sizeof pt);
    detail::put64(pt, m.txn);
    detail::put64(pt + 8, m.count);
    detail::put64(pt + 16, m.logical);
    std::memcpy(pt + 24, m.salt, kSalt);
    if (!fillRandom(out, kNonce)) return false;
    uint8_t aad[kAadLen];
    detail::makeAad(aad, 0, 0, kind, 0);
    crypto::xSeal(key_.data(), out, aad, kAadLen, pt, kMetaPlain, out + kNonce, out + kNonce + kMetaPlain);
    return true;
}

inline bool Engine::decodeMeta(uint32_t kind, const uint8_t* in, MetaRec& m) const {
    uint8_t aad[kAadLen];
    detail::makeAad(aad, 0, 0, kind, 0);
    uint8_t pt[kMetaPlain];
    ScopedWipe wPt(pt, sizeof pt);
    if (!crypto::xOpen(key_.data(), in, aad, kAadLen, in + kNonce, kMetaPlain,
                       in + kNonce + kMetaPlain, pt)) {
        return false;
    }
    m.txn = detail::get64(pt);
    m.count = detail::get64(pt + 8);
    m.logical = detail::get64(pt + 16);
    std::memcpy(m.salt, pt + 24, kSalt);
    return true;
}

inline Err Engine::open(const Config& in, std::unique_ptr<IBlockStore> store,
                        const uint8_t* key, size_t keyLen, std::unique_ptr<Engine>& out) noexcept {
    try {
        if (!store) return Err::BadConfig;
        if (!key || keyLen != kKey) return Err::BadKey;
        if (in.capacityBytes < kMinCapacity || in.capacityBytes > kMaxCapacity) return Err::BadConfig;
        if (in.chunkBytes < kMinChunk || in.chunkBytes > kMaxChunk || !detail::isPow2(in.chunkBytes)) {
            return Err::BadConfig;
        }
        if (!crypto::poly1305SelfTest()) return Err::Crypto;

        uint64_t storeSize = 0;
        if (!store->size(storeSize)) return Err::Io;

        std::unique_ptr<Engine> e(new Engine());
        e->key_ = SecureBytes(kKey);
        std::memcpy(e->key_.data(), key, kKey);
        e->store_ = std::move(store);
        e->requestedCache_ = in.cacheBytes;

        Super s;
        if (!e->loadBestSuper(s)) {
            // No authenticated superblock. Only an empty or metadata-sized file is safe to
            // initialize; anything larger may hold data we must not overwrite.
            if (storeSize > kMetaBytes) return Err::Format;
            s = Super{};
            s.gen = 1;
            s.capacity = in.capacityBytes;
            s.chunk = in.chunkBytes;
            e->setLayout(in.chunkBytes);
            if (e->writeSuperLocked(s) != Err::Ok) return Err::Io;
        } else {
            if (s.chunk < kMinChunk || s.chunk > kMaxChunk || !detail::isPow2(s.chunk)) return Err::Format;
            e->setLayout(s.chunk);
            if (s.capacity != in.capacityBytes) {
                if (in.capacityBytes < s.logical) return Err::BadConfig;
                Super t = s;
                t.gen = s.gen + 1;
                t.capacity = in.capacityBytes;
                if (e->writeSuperLocked(t) != Err::Ok) return Err::Io;
                s = t;
            }
            if (s.logical > 0) {
                const uint64_t required = e->slotOffset(detail::ceilDiv(s.logical, s.chunk));
                if (storeSize < required) return Err::Format;   // truncated below committed extent
            }
        }

        e->pool_ = std::make_shared<BufferPool>(e->chunkBytes_, kPoolIdle);
        e->open_ = true;
        {
            std::lock_guard<std::mutex> g(e->cfgMu_);
            e->applyCacheLocked();
        }
        if (const Err r = e->recoverLocked(); r != Err::Ok) return r;
        out = std::move(e);
        return Err::Ok;
    } catch (const std::bad_alloc&) {
        return Err::NoRam;
    } catch (...) {
        return Err::Io;
    }
}

// Verifies every entry of a committed transaction before any is applied. Position,
// transaction binding, and authentication must all hold.
inline Err Engine::verifyEntries(const uint8_t* entries, uint64_t n, const MetaRec& m) noexcept {
    try {
        const uint64_t cb = chunkBytes_;
        const uint64_t chunks = detail::ceilDiv(m.logical, cb);
        for (uint64_t i = 0; i < n; ++i) {
            const uint8_t* ent = entries + i * entryBytes_;
            const uint64_t idx = detail::get64(ent);
            const uint8_t* rec = ent + kEntryHdr;
            if (idx >= chunks) return Err::Format;
            if (detail::get64(rec + kSlotTxnOff) != m.txn) return Err::Format;

            auto scratch = pool_->acquire();
            uint8_t aad[kAadLen];
            detail::makeAad(aad, idx, chunkBytes_, kKindChunk, m.txn);
            if (!crypto::xOpen(key_.data(), rec, aad, kAadLen, rec + kSlotCtOff, cb,
                               rec + kSlotCtOff + cb, scratch->data())) {
                return Err::Crypto;
            }
        }
        return Err::Ok;
    } catch (const std::bad_alloc&) {
        return Err::NoRam;
    } catch (...) {
        return Err::Io;
    }
}

// Copies journal entries into data slots, fsyncs, then publishes via the superblock.
// Callers must hold the commit point (or have verified entries during recovery).
inline Err Engine::applyCommittedLocked(const uint8_t* entries, uint64_t n, const MetaRec& m) noexcept {
    try {
        for (uint64_t i = 0; i < n; ++i) {
            const uint8_t* ent = entries + i * entryBytes_;
            const uint64_t idx = detail::get64(ent);
            if (!store_->write(slotOffset(idx), ent + kEntryHdr, slotBytes_)) return Err::Io;
        }
        if (!store_->sync()) return Err::Io;

        Super s = sb_;
        s.gen = sb_.gen + 1;
        s.txn = m.txn;
        s.logical = m.logical;
        if (writeSuperLocked(s) != Err::Ok) return Err::Io;
        sb_ = s;
        committedTxn_.store(s.txn, std::memory_order_release);
        logical_.store(s.logical, std::memory_order_release);
        return Err::Ok;
    } catch (...) {
        return Err::Io;
    }
}

// Restores a consistent state from disk. Requires exclusive mu_ (or single-threaded open).
inline Err Engine::recoverLocked() noexcept {
    try {
        Super s;
        if (!loadBestSuper(s) || s.chunk != chunkBytes_) return Err::Format;
        sb_ = s;
        logical_.store(s.logical, std::memory_order_release);
        capacity_.store(s.capacity, std::memory_order_release);
        committedTxn_.store(s.txn, std::memory_order_release);
        trimCacheLocked();

        uint8_t beginRec[kMetaRec];
        MetaRec b;
        if (!store_->read(kMetaBytes, beginRec, kMetaRec) || !decodeMeta(kKindBegin, beginRec, b)) {
            return Err::Ok;   // no pending transaction
        }
        if (b.txn != s.txn + 1 || b.count == 0 || b.count > maxEntries_ ||
            b.logical < s.logical || b.logical > s.capacity) {
            return Err::Ok;   // stale journal from an earlier transaction
        }

        const uint64_t commitOff = kMetaBytes + kJrnBlock + b.count * entryBytes_;
        uint8_t commitRec[kMetaRec];
        MetaRec c;
        if (!store_->read(commitOff, commitRec, kMetaRec) || !decodeMeta(kKindCommit, commitRec, c)) {
            return Err::Ok;   // never reached the commit point: discard
        }
        if (c.txn != b.txn || c.count != b.count || c.logical != b.logical ||
            std::memcmp(c.salt, b.salt, kSalt) != 0) {
            return Err::Ok;   // commit belongs to a different attempt: discard
        }

        std::vector<uint8_t> entries(b.count * entryBytes_);
        if (!store_->read(kMetaBytes + kJrnBlock, entries.data(), entries.size())) return Err::Io;

        if (verifyEntries(entries.data(), b.count, b) != Err::Ok) {
            // Committed bytes failed verification. Nothing has been applied, so the vault
            // stays at the last published state.
            rolledBack_.fetch_add(1, std::memory_order_acq_rel);
            return Err::Ok;
        }
        const Err a = applyCommittedLocked(entries.data(), b.count, b);
        if (a == Err::Ok) rolledForward_.fetch_add(1, std::memory_order_acq_rel);
        return a;
    } catch (const std::bad_alloc&) {
        return Err::NoRam;
    } catch (...) {
        return Err::Io;
    }
}

// Roll forward from the journal when a previous operation faulted after its commit point.
inline Err Engine::healIfFaulted() noexcept {
    if (!faulted_.load(std::memory_order_acquire)) return Err::Ok;
    try {
        std::unique_lock<std::shared_mutex> session(mu_);
        if (!open_) return Err::NotOpen;
        if (!faulted_.load(std::memory_order_acquire)) return Err::Ok;
        const Err r = recoverLocked();
        if (r == Err::Ok) faulted_.store(false, std::memory_order_release);
        return r;
    } catch (...) {
        return Err::Io;
    }
}

// Loads and authenticates one chunk. Caller holds the chunk's stripe lock.
inline Err Engine::loadChunk(uint64_t idx, Chunk& out, bool cache) noexcept {
    try {
        if (cache) {
            out = cacheGet(idx);
            if (out) return Err::Ok;
        }
        auto buf = pool_->acquire();   // zero-filled
        if (idx * uint64_t(chunkBytes_) >= logical_.load(std::memory_order_acquire)) {
            out = std::move(buf);      // never-written region reads as zeros; not cached
            return Err::Ok;
        }

        thread_local std::vector<uint8_t> rec;
        rec.resize(static_cast<size_t>(slotBytes_));
        if (!store_->read(slotOffset(idx), rec.data(), rec.size())) return Err::Io;

        const uint64_t txn = detail::get64(rec.data() + kSlotTxnOff);
        if (txn > committedTxn_.load(std::memory_order_acquire)) return Err::Format;

        uint8_t aad[kAadLen];
        detail::makeAad(aad, idx, chunkBytes_, kKindChunk, txn);
        if (!crypto::xOpen(key_.data(), rec.data(), aad, kAadLen,
                           rec.data() + kSlotCtOff, chunkBytes_,
                           rec.data() + kSlotCtOff + chunkBytes_, buf->data())) {
            return Err::Crypto;
        }
        if (cache) cachePut(idx, buf);
        out = std::move(buf);
        return Err::Ok;
    } catch (const std::bad_alloc&) {
        return Err::NoRam;
    } catch (...) {
        return Err::Io;
    }
}

// Transactional write. Nothing is visible until the superblock publishes it.
inline Err Engine::write(uint64_t off, const uint8_t* data, uint64_t len) noexcept {
    bool committed = false;
    try {
        if (len == 0) return Err::Ok;
        if (!data) return Err::BadArg;
        if (len > kMaxIoPerCall) return Err::TooLarge;
        if (const Err h = healIfFaulted(); h != Err::Ok) return h;

        std::shared_lock<std::shared_mutex> session(mu_);
        if (!open_) return Err::NotOpen;
        if (faulted_.load(std::memory_order_acquire)) return Err::Io;

        std::lock_guard<std::mutex> txnLock(txnMu_);
        const uint64_t cb = chunkBytes_;
        const uint64_t cap = capacity_.load(std::memory_order_acquire);
        const uint64_t logical = sb_.logical;
        if (off > logical || off > cap || len > cap - off) return Err::OutOfRange;

        const uint64_t end = off + len;
        const uint64_t newLogical = (std::max)(logical, end);
        const uint64_t first = off / cb;
        const uint64_t last = (end - 1) / cb;
        const uint64_t count = last - first + 1;

        // Space check covers data growth, the journal footprint, and a safety margin.
        uint64_t grow = 0;
        if (newLogical > logical) {
            grow = (detail::ceilDiv(newLogical, cb) - detail::ceilDiv(logical, cb)) * slotBytes_;
        }
        const uint64_t journalNeed = kJrnBlock + count * entryBytes_ + kMetaRec;
        const uint64_t need = grow + grow / 20 + journalNeed + kMinFreeAfterGrowth;
        uint64_t avail = 0;
        if (!store_->freeSpace(avail)) return Err::Io;
        if (avail < need) return Err::NoSpace;

        StripeSet stripes(stripes_, first, last, /*exclusive=*/true);

        MetaRec begin;
        begin.txn = sb_.txn + 1;
        begin.count = count;
        begin.logical = newLogical;
        if (!fillRandom(begin.salt, kSalt)) return Err::Crypto;

        // Build encrypted journal entries. Ciphertext only leaves this scope.
        std::vector<uint8_t> entries(count * entryBytes_);
        std::vector<Chunk> images;
        images.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i) {
            const uint64_t idx = first + i;
            const uint64_t chunkStart = idx * cb;
            const uint64_t segStart = (std::max)(off, chunkStart);
            const uint64_t segEnd = (std::min)(end, chunkStart + cb);
            const uint64_t within = segStart - chunkStart;
            const uint64_t take = segEnd - segStart;

            Chunk img = pool_->acquire();
            if (take != cb) {
                Chunk old;
                const Err r = loadChunk(idx, old, /*cache=*/true);
                if (r != Err::Ok) return r;
                std::memcpy(img->data(), old->data(), static_cast<size_t>(cb));
            }
            std::memcpy(img->data() + within, data + (segStart - off), static_cast<size_t>(take));

            uint8_t* ent = entries.data() + i * entryBytes_;
            detail::put64(ent, idx);
            uint8_t* rec = ent + kEntryHdr;
            if (!fillRandom(rec, kNonce)) return Err::Crypto;
            detail::put64(rec + kSlotTxnOff, begin.txn);
            uint8_t aad[kAadLen];
            detail::makeAad(aad, idx, chunkBytes_, kKindChunk, begin.txn);
            crypto::xSeal(key_.data(), rec, aad, kAadLen, img->data(), static_cast<size_t>(cb),
                          rec + kSlotCtOff, rec + kSlotCtOff + cb);
            images.push_back(std::move(img));
        }

        // Stage 1: announce the attempt. The fresh begin record makes any commit record
        // left from an earlier attempt at this txn fail its salt check, so entries that
        // are partially overwritten below can never be applied under that old commit.
        uint8_t beginRec[kMetaRec];
        if (!encodeMeta(kKindBegin, begin, beginRec)) return Err::Crypto;
        if (!store_->write(kMetaBytes, beginRec, kMetaRec)) return Err::Io;
        if (!store_->sync()) return Err::Io;

        // Stage 2: journal payload, durable before the commit record.
        if (!store_->write(kMetaBytes + kJrnBlock, entries.data(), entries.size())) return Err::Io;
        if (!store_->sync()) return Err::Io;

        // Commit point. A failure here is an unknown outcome, so the engine must heal.
        uint8_t commitRec[kMetaRec];
        if (!encodeMeta(kKindCommit, begin, commitRec)) return Err::Crypto;
        const uint64_t commitOff = kMetaBytes + kJrnBlock + count * entryBytes_;
        if (!store_->write(commitOff, commitRec, kMetaRec) || !store_->sync()) {
            faulted_.store(true, std::memory_order_release);
            return Err::Io;
        }
        committed = true;

        // Stage 3: apply and publish. Failure here rolls forward on the next operation.
        if (applyCommittedLocked(entries.data(), count, begin) != Err::Ok) {
            faulted_.store(true, std::memory_order_release);
            return Err::Io;
        }
        for (uint64_t i = 0; i < count; ++i) cachePut(first + i, images[static_cast<size_t>(i)]);
        return Err::Ok;
    } catch (const std::bad_alloc&) {
        if (committed) faulted_.store(true, std::memory_order_release);
        return Err::NoRam;
    } catch (...) {
        if (committed) faulted_.store(true, std::memory_order_release);
        return Err::Io;
    }
}

inline Err Engine::read(uint64_t off, uint8_t* out, uint64_t len) noexcept {
    try {
        if (len == 0) return Err::Ok;
        if (!out) return Err::BadArg;
        if (len > kMaxIoPerCall) return Err::TooLarge;
        if (const Err h = healIfFaulted(); h != Err::Ok) return h;

        std::shared_lock<std::shared_mutex> session(mu_);
        if (!open_) return Err::NotOpen;
        if (faulted_.load(std::memory_order_acquire)) return Err::Io;
        const uint64_t logical = logical_.load(std::memory_order_acquire);
        if (off > logical || len > logical - off) return Err::OutOfRange;

        const uint64_t cb = chunkBytes_;
        const uint64_t first = off / cb;
        const uint64_t last = (off + len - 1) / cb;
        StripeSet stripes(stripes_, first, last, /*exclusive=*/false);

        uint64_t done = 0;
        while (done < len) {
            const uint64_t pos = off + done;
            const uint64_t idx = pos / cb;
            const uint64_t within = pos % cb;
            const uint64_t take = (std::min<uint64_t>)(cb - within, len - done);
            Chunk c;
            const Err e = loadChunk(idx, c, /*cache=*/true);
            if (e != Err::Ok) {
                secureWipe(out, static_cast<size_t>(len));
                return e;
            }
            std::memcpy(out + done, c->data() + within, static_cast<size_t>(take));
            done += take;
        }
        return Err::Ok;
    } catch (const std::bad_alloc&) {
        return Err::NoRam;
    } catch (...) {
        return Err::Io;
    }
}

// Authenticates every committed chunk without caching plaintext.
inline Err Engine::scrub(uint64_t& bad) noexcept {
    bad = 0;
    try {
        if (const Err h = healIfFaulted(); h != Err::Ok) return h;

        std::shared_lock<std::shared_mutex> session(mu_);
        if (!open_) return Err::NotOpen;
        if (faulted_.load(std::memory_order_acquire)) return Err::Io;

        std::lock_guard<std::mutex> txnLock(txnMu_);
        const uint64_t chunks = detail::ceilDiv(sb_.logical, chunkBytes_);
        for (uint64_t idx = 0; idx < chunks; ++idx) {
            StripeSet stripe(stripes_, idx, idx, /*exclusive=*/false);
            Chunk c;
            const Err e = loadChunk(idx, c, /*cache=*/false);
            if (e == Err::Crypto || e == Err::Format) {
                ++bad;
            } else if (e != Err::Ok) {
                return e;
            }
        }
        return Err::Ok;
    } catch (const std::bad_alloc&) {
        return Err::NoRam;
    } catch (...) {
        return Err::Io;
    }
}

inline Err Engine::health(Health& h) const noexcept {
    h.committedTxn = committedTxn_.load(std::memory_order_acquire);
    h.rolledForward = rolledForward_.load(std::memory_order_acquire);
    h.rolledBack = rolledBack_.load(std::memory_order_acquire);
    h.faulted = faulted_.load(std::memory_order_acquire);
    return Err::Ok;
}

inline Err Engine::resize(uint64_t newCap) noexcept {
    try {
        if (newCap < kMinCapacity || newCap > kMaxCapacity) return Err::BadConfig;
        if (const Err h = healIfFaulted(); h != Err::Ok) return h;

        std::unique_lock<std::shared_mutex> session(mu_);
        if (!open_) return Err::NotOpen;
        if (faulted_.load(std::memory_order_acquire)) return Err::Io;
        if (newCap < sb_.logical) return Err::OutOfRange;

        Super s = sb_;
        s.gen = sb_.gen + 1;
        s.capacity = newCap;
        if (writeSuperLocked(s) != Err::Ok) {
            faulted_.store(true, std::memory_order_release);
            return Err::Io;
        }
        sb_ = s;
        capacity_.store(newCap, std::memory_order_release);
        return Err::Ok;
    } catch (const std::bad_alloc&) {
        return Err::NoRam;
    } catch (...) {
        return Err::Io;
    }
}

inline Err Engine::destroy() noexcept {
    try {
        std::unique_lock<std::shared_mutex> session(mu_);
        if (!open_) return Err::NotOpen;
        open_ = false;
        faulted_.store(false, std::memory_order_release);
        logical_.store(0, std::memory_order_release);
        committedTxn_.store(0, std::memory_order_release);
        trimCacheLocked();
        key_.release();
        const bool ok = store_ && store_->erase();
        return ok ? Err::Ok : Err::Io;
    } catch (...) {
        return Err::Io;
    }
}

inline void Engine::close() noexcept {
    try {
        std::unique_lock<std::shared_mutex> session(mu_);
        if (store_) {
            if (open_) store_->sync();
            store_->close();
        }
        open_ = false;
        trimCacheLocked();
        key_.release();
    } catch (...) {
        key_.release();
    }
}

inline void Engine::eraseIndexLocked(CacheShard& s, uint64_t idx) noexcept {
    auto it = s.index.find(idx);
    if (it == s.index.end()) return;
    s.lru.erase(it->second);
    s.index.erase(it);
}

inline void Engine::evictShardLocked(CacheShard& s) noexcept {
    while (s.lru.size() > s.maxEntries) {
        s.index.erase(s.lru.back().first);
        s.lru.pop_back();   // last reference returns and zeroizes the buffer
    }
}

inline void Engine::applyCacheLocked() {
    const uint64_t budget = clampCache(requestedCache_);
    const uint64_t total = budget / chunkBytes_;
    for (size_t i = 0; i < kCacheShards; ++i) {
        CacheShard& s = shards_[i];
        std::lock_guard<std::mutex> g(s.mu);
        s.maxEntries = size_t(total / kCacheShards + (i < total % kCacheShards ? 1 : 0));
        evictShardLocked(s);
    }
}

inline void Engine::setCacheBytes(uint64_t requested) noexcept {
    try {
        std::lock_guard<std::mutex> g(cfgMu_);
        requestedCache_ = requested;
        applyCacheLocked();
    } catch (...) {}
}

inline void Engine::trimCacheLocked() noexcept {
    for (auto& s : shards_) {
        try {
            std::lock_guard<std::mutex> g(s.mu);
            s.index.clear();
            s.lru.clear();
        } catch (...) {}
    }
}

inline Engine::Chunk Engine::cacheGet(uint64_t idx) noexcept {
    CacheShard& s = shards_[idx % kCacheShards];
    try {
        std::lock_guard<std::mutex> g(s.mu);
        auto it = s.index.find(idx);
        if (it == s.index.end()) return nullptr;
        s.lru.splice(s.lru.begin(), s.lru, it->second);
        return it->second->second;
    } catch (...) {
        return nullptr;
    }
}

// If an entry cannot be cached, any previous entry for this index is removed so stale
// plaintext is never served.
inline void Engine::cachePut(uint64_t idx, Chunk p) noexcept {
    CacheShard& s = shards_[idx % kCacheShards];
    try {
        std::lock_guard<std::mutex> g(s.mu);
        if (s.maxEntries == 0) {
            eraseIndexLocked(s, idx);
            return;
        }
        auto it = s.index.find(idx);
        if (it != s.index.end()) {
            it->second->second = std::move(p);
            s.lru.splice(s.lru.begin(), s.lru, it->second);
            return;
        }
        s.lru.emplace_front(idx, std::move(p));
        try {
            s.index.emplace(idx, s.lru.begin());
        } catch (...) {
            s.lru.pop_front();
            throw;
        }
        evictShardLocked(s);
    } catch (...) {
        cacheErase(idx);
    }
}

inline void Engine::cacheErase(uint64_t idx) noexcept {
    CacheShard& s = shards_[idx % kCacheShards];
    try {
        std::lock_guard<std::mutex> g(s.mu);
        eraseIndexLocked(s, idx);
    } catch (...) {}
}
} // namespace sd

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------

#define SD_STATUS_LIST(X) \
    X(SD_OK, 0) X(SD_ERR_NOT_OPEN, 1) X(SD_ERR_BAD_CONFIG, 2) X(SD_ERR_BAD_KEY, 3) \
    X(SD_ERR_BAD_ARG, 4) X(SD_ERR_OUT_OF_RANGE, 5) X(SD_ERR_TOO_LARGE, 6) X(SD_ERR_NO_SPACE, 7) \
    X(SD_ERR_NO_RAM, 8) X(SD_ERR_IO, 9) X(SD_ERR_CRYPTO, 10) X(SD_ERR_FORMAT, 11)

typedef enum sd_status {
#define SD_DECL_STATUS(name, val) name = val,
    SD_STATUS_LIST(SD_DECL_STATUS)
#undef SD_DECL_STATUS
} sd_status;

#define SD_FLAG_HARDEN_PROCESS 0x1u

typedef struct sd_config_c {
    uint64_t capacity_bytes;
    uint64_t cache_bytes;
    uint32_t chunk_bytes;
    uint32_t flags;        // zero-initialize for default behavior
} sd_config_c;

typedef struct sd_health_c {
    uint64_t committed_txn;
    uint64_t rolled_forward;
    uint64_t rolled_back;
    uint32_t faulted;
    uint32_t reserved;
} sd_health_c;

namespace sd_detail {

constexpr size_t kRegShards = 32;

struct RegShard {
    std::shared_mutex mu;
    std::unordered_map<sd_handle, std::shared_ptr<sd::Engine>> map;
};

struct Registry {
    std::array<RegShard, kRegShards> shards;
    std::atomic<uint64_t> nextId{1};
};

inline Registry& registry() noexcept {
    static Registry r;
    return r;
}

inline RegShard& shardFor(sd_handle h) noexcept {
    return registry().shards[static_cast<size_t>(h % kRegShards)];
}

inline sd_status toStatus(sd::Err e) noexcept {
    switch (e) {
        case sd::Err::Ok: return SD_OK;
        case sd::Err::NotOpen: return SD_ERR_NOT_OPEN;
        case sd::Err::BadConfig: return SD_ERR_BAD_CONFIG;
        case sd::Err::BadKey: return SD_ERR_BAD_KEY;
        case sd::Err::BadArg: return SD_ERR_BAD_ARG;
        case sd::Err::OutOfRange: return SD_ERR_OUT_OF_RANGE;
        case sd::Err::TooLarge: return SD_ERR_TOO_LARGE;
        case sd::Err::NoSpace: return SD_ERR_NO_SPACE;
        case sd::Err::NoRam: return SD_ERR_NO_RAM;
        case sd::Err::Io: return SD_ERR_IO;
        case sd::Err::Crypto: return SD_ERR_CRYPTO;
        case sd::Err::Format: return SD_ERR_FORMAT;
    }
    return SD_ERR_IO;
}

inline sd::Config toConfig(const sd_config_c& c) {
    sd::Config cfg;
    cfg.capacityBytes = c.capacity_bytes;
    cfg.cacheBytes = c.cache_bytes ? c.cache_bytes : sd::kDefaultCache;
    cfg.chunkBytes = c.chunk_bytes ? c.chunk_bytes : sd::kDefaultChunk;
    return cfg;
}

inline bool boundedCString(const char* s, size_t maxLen) noexcept {
    if (!s) return false;
    return ::strnlen(s, maxLen + 1) <= maxLen;
}

inline std::shared_ptr<sd::Engine> lookup(sd_handle h) noexcept {
    if (h == 0) return nullptr;
    try {
        RegShard& s = shardFor(h);
        std::shared_lock<std::shared_mutex> lk(s.mu);
        auto it = s.map.find(h);
        return it == s.map.end() ? nullptr : it->second;
    } catch (...) {
        return nullptr;
    }
}

template <typename Fn>
inline sd_status runStatus(sd_handle h, Fn&& fn) noexcept {
    try {
        std::shared_ptr<sd::Engine> e = lookup(h);
        if (!e) return SD_ERR_NOT_OPEN;
        return toStatus(fn(*e));
    } catch (const std::bad_alloc&) {
        return SD_ERR_NO_RAM;
    } catch (...) {
        return SD_ERR_IO;
    }
}

inline sd_status publish(std::unique_ptr<sd::Engine> eng, sd_handle* out) {
    const sd_handle h = registry().nextId.fetch_add(1, std::memory_order_relaxed);
    if (h == 0) return SD_ERR_IO;
    try {
        RegShard& s = shardFor(h);
        std::unique_lock<std::shared_mutex> lk(s.mu);
        s.map.emplace(h, std::shared_ptr<sd::Engine>(std::move(eng)));
    } catch (const std::bad_alloc&) {
        return SD_ERR_NO_RAM;
    } catch (...) {
        return SD_ERR_IO;
    }
    *out = h;
    return SD_OK;
}

} // namespace sd_detail

SD_API uint32_t sd_api_version(void) { return 0x00080000u; }

SD_API const char* sd_status_string(sd_status s) {
    switch (s) {
#define SD_CASE_NAME(name, val) case name: return #name;
        SD_STATUS_LIST(SD_CASE_NAME)
#undef SD_CASE_NAME
    }
    return "SD_ERR_UNKNOWN";
}

SD_API sd_status sd_open_file(const char* dir, const sd_config_c* cfg, const uint8_t* key32, sd_handle* out) {
    if (!out) return SD_ERR_BAD_ARG;
    *out = 0;
    if (!cfg || !key32 || !sd_detail::boundedCString(dir, sd::kMaxPathLen)) return SD_ERR_BAD_ARG;
    if ((cfg->flags & ~SD_FLAG_HARDEN_PROCESS) != 0) return SD_ERR_BAD_CONFIG;
    if (cfg->flags & SD_FLAG_HARDEN_PROCESS) sd::hardenProcess();
    try {
        std::unique_ptr<sd::IBlockStore> store = sd::FileBlockStore::create(dir);
        if (!store) return SD_ERR_IO;
        std::unique_ptr<sd::Engine> eng;
        const sd::Err e = sd::Engine::open(sd_detail::toConfig(*cfg), std::move(store), key32, sd::kKey, eng);
        if (e != sd::Err::Ok) return sd_detail::toStatus(e);
        return sd_detail::publish(std::move(eng), out);
    } catch (const std::bad_alloc&) {
        return SD_ERR_NO_RAM;
    } catch (...) {
        return SD_ERR_IO;
    }
}

SD_API void sd_close(sd_handle h) {
    try {
        std::shared_ptr<sd::Engine> e;
        {
            sd_detail::RegShard& s = sd_detail::shardFor(h);
            std::unique_lock<std::shared_mutex> lk(s.mu);
            auto it = s.map.find(h);
            if (it == s.map.end()) return;
            e = std::move(it->second);
            s.map.erase(it);
        }
        e->close();
    } catch (...) {}
}

SD_API sd_status sd_destroy(sd_handle h) {
    return sd_detail::runStatus(h, [](sd::Engine& x) { return x.destroy(); });
}

SD_API sd_status sd_write(sd_handle h, uint64_t offset, const uint8_t* data, uint64_t len) {
    return sd_detail::runStatus(h, [&](sd::Engine& x) { return x.write(offset, data, len); });
}

SD_API sd_status sd_read_bounded(sd_handle h, uint64_t offset, uint8_t* out, size_t out_cap, uint64_t len) {
    if (!out || len > static_cast<uint64_t>(out_cap)) return SD_ERR_BAD_ARG;
    return sd_detail::runStatus(h, [&](sd::Engine& x) { return x.read(offset, out, len); });
}

// Authenticates every committed chunk. Returns the number of chunks that failed.
SD_API sd_status sd_scrub(sd_handle h, uint64_t* bad_chunks) {
    if (!bad_chunks) return SD_ERR_BAD_ARG;
    *bad_chunks = 0;
    return sd_detail::runStatus(h, [&](sd::Engine& x) { return x.scrub(*bad_chunks); });
}

SD_API sd_status sd_health(sd_handle h, sd_health_c* out) {
    if (!out) return SD_ERR_BAD_ARG;
    sd::Engine::Health hh{};
    const sd_status s = sd_detail::runStatus(h, [&](sd::Engine& x) { return x.health(hh); });
    if (s == SD_OK) {
        out->committed_txn = hh.committedTxn;
        out->rolled_forward = hh.rolledForward;
        out->rolled_back = hh.rolledBack;
        out->faulted = hh.faulted ? 1u : 0u;
        out->reserved = 0;
    }
    return s;
}

// ---------------------------------------------------------------------------
// Android JNI. Method names and signatures are unchanged. Do not add native methods
// here without matching Java declarations: RegisterNatives fails on unknown methods.
// ---------------------------------------------------------------------------

#if defined(SD_ENABLE_JNI)
#include <jni.h>
namespace sd_jni {
constexpr const char* kClass = "com/zeeshan/audiodiary/NativeVault";

inline jlong jOpen(JNIEnv* env, jobject, jstring jDir, jbyteArray jKey, jlong capacity) noexcept {
    if (!env || !jDir || !jKey || capacity < 0) return 0;
    if (env->GetArrayLength(jKey) != static_cast<jsize>(sd::kKey)) return 0;

    const char* dir = env->GetStringUTFChars(jDir, nullptr);
    if (!dir) return 0;

    sd_handle h = 0;
    uint8_t key[sd::kKey] = {0};
    sd::ScopedWipe wKey(key, sizeof key);
    env->GetByteArrayRegion(jKey, 0, static_cast<jsize>(sd::kKey), reinterpret_cast<jbyte*>(key));
    if (!env->ExceptionCheck()) {
        sd_config_c cfg{};
        cfg.capacity_bytes = static_cast<uint64_t>(capacity);
        cfg.flags = SD_FLAG_HARDEN_PROCESS;
        if (sd_open_file(dir, &cfg, key, &h) != SD_OK) h = 0;
    }
    env->ReleaseStringUTFChars(jDir, dir);
    return static_cast<jlong>(h);
}

inline jint jWrite(JNIEnv* env, jobject, jlong h, jlong off, jbyteArray data) noexcept {
    if (!env || !data || h <= 0 || off < 0) return SD_ERR_BAD_ARG;
    const jsize n = env->GetArrayLength(data);
    if (n < 0 || static_cast<uint64_t>(n) > sd::kMaxIoPerCall) return SD_ERR_TOO_LARGE;
    if (n == 0) return SD_OK;

    jint status = SD_ERR_IO;
    try {
        sd::SecureBytes buf(static_cast<size_t>(n));
        env->GetByteArrayRegion(data, 0, n, reinterpret_cast<jbyte*>(buf.data()));
        if (!env->ExceptionCheck()) {
            status = sd_write(static_cast<sd_handle>(h), static_cast<uint64_t>(off),
                              buf.data(), static_cast<uint64_t>(n));
        }
    } catch (const std::bad_alloc&) {
        status = SD_ERR_NO_RAM;
    } catch (...) {
        status = SD_ERR_IO;
    }
    return status;
}

inline jbyteArray jRead(JNIEnv* env, jobject, jlong h, jlong off, jlong len) noexcept {
    if (!env || h <= 0 || off < 0 || len <= 0) return nullptr;
    if (static_cast<uint64_t>(len) > sd::kMaxIoPerCall) return nullptr;
    try {
        sd::SecureBytes buf(static_cast<size_t>(len));
        if (sd_read_bounded(static_cast<sd_handle>(h), static_cast<uint64_t>(off),
                            buf.data(), buf.size(), static_cast<uint64_t>(len)) != SD_OK) {
            return nullptr;
        }
        jbyteArray out = env->NewByteArray(static_cast<jsize>(buf.size()));
        if (!out) return nullptr;
        env->SetByteArrayRegion(out, 0, static_cast<jsize>(buf.size()), reinterpret_cast<const jbyte*>(buf.data()));
        if (env->ExceptionCheck()) {
            env->DeleteLocalRef(out);
            return nullptr;
        }
        return out;
    } catch (...) {
        return nullptr;
    }
}

inline void jClose(JNIEnv*, jobject, jlong h) noexcept {
    if (h <= 0) return;
    sd_close(static_cast<sd_handle>(h));
}

inline jint jDestroy(JNIEnv*, jobject, jlong h) noexcept {
    if (h <= 0) return SD_ERR_NOT_OPEN;
    return sd_destroy(static_cast<sd_handle>(h));
}
} // namespace sd_jni

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
    if (!vm) return JNI_ERR;
    JNIEnv* env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK || !env) return JNI_ERR;
    jclass cls = env->FindClass(sd_jni::kClass);
    if (!cls) {
        env->ExceptionClear();
        return JNI_ERR;
    }
    static const JNINativeMethod methods[] = {
        {const_cast<char*>("nativeOpenFile"), const_cast<char*>("(Ljava/lang/String;[BJ)J"),
         reinterpret_cast<void*>(&sd_jni::jOpen)},
        {const_cast<char*>("nativeWrite"), const_cast<char*>("(JJ[B)I"),
         reinterpret_cast<void*>(&sd_jni::jWrite)},
        {const_cast<char*>("nativeRead"), const_cast<char*>("(JJJ)[B"),
         reinterpret_cast<void*>(&sd_jni::jRead)},
        {const_cast<char*>("nativeDestroy"), const_cast<char*>("(J)I"),
         reinterpret_cast<void*>(&sd_jni::jDestroy)},
        {const_cast<char*>("nativeClose"), const_cast<char*>("(J)V"),
         reinterpret_cast<void*>(&sd_jni::jClose)},
    };
    const jint rc = env->RegisterNatives(cls, methods, static_cast<jint>(sizeof methods / sizeof methods[0]));
    env->DeleteLocalRef(cls);
    if (rc != JNI_OK) {
        env->ExceptionClear();
        return JNI_ERR;
    }
    return JNI_VERSION_1_6;
}
#endif
