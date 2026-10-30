// ---------------------------------------------------------------------------
// cachesweep.cc - multi-configuration D-cache design-space simulator for Spike
//
// Loadable extlib (no Spike rebuild), used together with the qmac8 extension:
//
//   spike --isa=rv64imc_zicntr --extlib=ext/libqmac8.so --extlib=ext/libcachesweep.so
//         --extension=qmac8 --extension=cachesweep  prog.elf
//
// ---------------------------------------------------------------------------
#include "extension.h"
#include "mmu.h"
#include "memtracer.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int NPH = 64;
constexpr int PH_IGNORE = 63;

struct Stats {
  uint64_t rd = 0, wr = 0, rdm = 0, wrm = 0, wb = 0;
};

// SRAND reproduces Spike's own cache_sim_t exactly (random way even when an
// invalid way exists); it exists only to validate this simulator against `--dc`.
enum Policy { LRU, PLRU, RAND, FIFO, SRAND };

class Cache {
 public:
  std::string name;
  size_t size, ways, line, sets;
  int lshift, levels;
  Policy pol;
  std::vector<uint64_t> tag;     // full line number
  std::vector<uint8_t> valid, dirty;
  std::vector<uint32_t> stamp;   // LRU: last-use time
  std::vector<uint32_t> tree;    // PLRU: ways-1 bits per set
  std::vector<uint32_t> fifo;    // FIFO: next victim
  uint32_t lfsr = 1;
  uint32_t now = 0;
  Stats st[NPH];

  Cache(const std::string& n, size_t sz, size_t w, size_t ln, Policy p)
      : name(n), size(sz), ways(w), line(ln), pol(p) {
    sets = size / (ways * line);
    lshift = 0;
    while ((size_t(1) << lshift) < line) lshift++;
    levels = 0;
    while ((size_t(1) << levels) < ways) levels++;
    tag.assign(sets * ways, 0);
    valid.assign(sets * ways, 0);
    dirty.assign(sets * ways, 0);
    stamp.assign(sets * ways, 0);
    tree.assign(sets, 0);
    fifo.assign(sets, 0);
  }

  uint32_t rnd() { return lfsr = (lfsr >> 1) ^ (-(lfsr & 1) & 0xd0000001); }

  // replacement-state update on a hit or a fill of `way`
  void touch(size_t set, size_t way) {
    if (pol == LRU) {
      stamp[set * ways + way] = ++now;
    } else if (pol == PLRU && ways > 1) {
      uint32_t node = 0;
      uint32_t& t = tree[set];
      for (int l = levels - 1; l >= 0; l--) {
        uint32_t bit = (way >> l) & 1;
        if (bit) t &= ~(1u << node); else t |= (1u << node);   // point AWAY from way
        node = 2 * node + 1 + bit;
      }
    }
  }

  size_t victim(size_t set) {
    size_t base = set * ways;
    if (pol == SRAND) return rnd() % ways;
    for (size_t w = 0; w < ways; w++)
      if (!valid[base + w]) return w;                          // invalid way first
    switch (pol) {
      case LRU: {
        size_t b = 0;
        for (size_t w = 1; w < ways; w++)
          if (stamp[base + w] < stamp[base + b]) b = w;
        return b;
      }
      case PLRU: {
        if (ways == 1) return 0;
        uint32_t node = 0, way = 0;
        for (int l = 0; l < levels; l++) {
          uint32_t bit = (tree[set] >> node) & 1;
          way = (way << 1) | bit;
          node = 2 * node + 1 + bit;
        }
        return way;
      }
      case FIFO: {
        size_t w = fifo[set];
        fifo[set] = (fifo[set] + 1) % ways;
        return w;
      }
      default:
        return rnd() % ways;
    }
  }

  void access(uint64_t addr, bool store, int ph) {
    uint64_t ln = addr >> lshift;
    size_t set = ln & (sets - 1);
    size_t base = set * ways;
    Stats& s = st[ph];
    store ? s.wr++ : s.rd++;
    for (size_t w = 0; w < ways; w++) {
      if (valid[base + w] && tag[base + w] == ln) {
        if (store) dirty[base + w] = 1;
        touch(set, w);
        return;
      }
    }
    store ? s.wrm++ : s.rdm++;
    size_t v = victim(set);
    if (valid[base + v] && dirty[base + v]) s.wb++;
    valid[base + v] = 1;
    tag[base + v] = ln;
    dirty[base + v] = store ? 1 : 0;
    touch(set, v);
  }
};

class sweep_tracer_t : public memtracer_t {
 public:
  std::vector<Cache*> caches;
  uint64_t mark = 0;
  int phase = PH_IGNORE;

  bool interested_in_range(uint64_t, uint64_t, access_type t) override {
    return t == LOAD || t == STORE;
  }

  void trace(uint64_t addr, size_t bytes, access_type t) override {
    if (t == FETCH) return;
    bool store = (t == STORE);
    if (store && mark && addr >= mark && addr < mark + NPH) {
      phase = (int)(addr - mark);
      return;
    }
    if (phase == PH_IGNORE) return;
    if (bytes == 0) bytes = 1;
    for (Cache* c : caches) {
      uint64_t first = addr >> c->lshift, last = (addr + bytes - 1) >> c->lshift;
      for (uint64_t l = first; l <= last; l++)
        c->access(l << c->lshift, store, phase);
    }
  }

  void clean_invalidate(uint64_t, size_t, bool, bool) override {}
};

sweep_tracer_t g_tracer;

bool pow2(unsigned long x) { return x && !(x & (x - 1)); }

void parse_configs() {
  const char* e = getenv("CS_CONFIGS");
  if (!e) return;
  std::string s(e);
  size_t p = 0;
  while (p < s.size()) {
    size_t q = s.find(';', p);
    if (q == std::string::npos) q = s.size();
    std::string item = s.substr(p, q - p);
    p = q + 1;
    if (item.empty()) continue;
    char nm[128], po[16];
    unsigned long sz, w, ln;
    if (sscanf(item.c_str(), "%127[^,],%lu,%lu,%lu,%15s", nm, &sz, &w, &ln, po) != 5) {
      fprintf(stderr, "cachesweep: bad config '%s'\n", item.c_str());
      exit(2);
    }
    Policy pol = LRU;
    if (!strcmp(po, "plru")) pol = PLRU;
    else if (!strcmp(po, "rand")) pol = RAND;
    else if (!strcmp(po, "fifo")) pol = FIFO;
    else if (!strcmp(po, "srand")) pol = SRAND;
    else if (strcmp(po, "lru")) {
      fprintf(stderr, "cachesweep: unknown policy '%s'\n", po);
      exit(2);
    }
    if (ln < 8 || !pow2(ln) || !w || sz % (w * ln) || !pow2(sz / (w * ln)) || (pol == PLRU && !pow2(w))) {
      fprintf(stderr, "cachesweep: invalid geometry '%s'\n", item.c_str());
      exit(2);
    }
    g_tracer.caches.push_back(new Cache(nm, sz, w, ln, pol));
  }
}

void dump() {
  const char* o = getenv("CS_OUT");
  FILE* f = o ? fopen(o, "w") : stderr;
  if (!f) f = stderr;
  for (Cache* c : g_tracer.caches)
    for (int p = 0; p < NPH; p++) {
      const Stats& s = c->st[p];
      if (s.rd + s.wr)
        fprintf(f, "CS %s %d %lu %lu %lu %lu %lu\n", c->name.c_str(), p,
                (unsigned long)s.rd, (unsigned long)s.wr, (unsigned long)s.rdm,
                (unsigned long)s.wrm, (unsigned long)s.wb);
    }
  if (f != stderr) fclose(f);
}

class cachesweep_ext_t : public extension_t {
 public:
  ~cachesweep_ext_t() override { dump(); }
  const char* name() const override { return "cachesweep"; }

  std::vector<insn_desc_t> get_instructions(const processor_t& proc) override {
    if (!installed) {                       // called once at registration
      installed = true;
      const char* m = getenv("CS_MARK");
      g_tracer.mark = m ? strtoull(m, nullptr, 16) : 0;
      if (!g_tracer.mark) g_tracer.phase = 0;     // no markers: everything is phase 0
      parse_configs();
      const_cast<processor_t&>(proc).get_mmu()->register_memtracer(&g_tracer);
    }
    return {};
  }
  std::vector<disasm_insn_t*> get_disasms(const processor_t*) override { return {}; }

 private:
  bool installed = false;
};

}  // namespace

REGISTER_EXTENSION(cachesweep, []() { static cachesweep_ext_t ext; return &ext; })

