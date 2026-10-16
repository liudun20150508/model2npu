// MTX-NPU v0 指令级仿真器 (ISS) —— 把量化权重 + 微码跑成"硬件运行结果"。
//
// 定位: 这不是 RTL, 是**周期模型 + 功能模型**二合一:
//   * 功能上按硬件数据排布真实算一遍 (int4 解包/去量化/分组累加/fp16 激活存储),
//     所以输出可以直接和 PyTorch/参考算子对拍;
//   * 时序上按 roofline 记 cycle (算力 / DRAM 带宽 / 固定开销), 所以能出
//     TTFT / TPOT / 带宽利用率 / 每算子瓶颈归因。
//
// 编程模型: 声明式张量 + 微码指令 (见 gen_program.py 生成)
//   ACT/KVC 声明激活与 KV (DRAM 驻留), IMAGE 指向 s3_pack.py 打出的设备镜像,
//   指令形如 GEMM out=x, a=ln1, w=w.L0.q, bias=b.L0.q
//
// 用法:
//   npu_iss --program p.mtx --image image.bin --idx image.idx --out results/npu_profile.json
//   npu_iss --dump-dequant build/deq --image ... --idx ...      # 权重格式 bit 级验证
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using std::string;
using std::vector;
using f32 = float;

// ------------------------------------------------------------------ 基础
static void die(const string &m) { fprintf(stderr, "[ISS] %s\n", m.c_str()); exit(2); }

static uint16_t f32_to_f16(f32 x) {
    uint32_t u; memcpy(&u, &x, 4);
    uint32_t sign = (u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((u >> 23) & 0xFF) - 127 + 15;
    uint32_t man = u & 0x7FFFFFu;
    if (((u >> 23) & 0xFF) == 0xFF) return (uint16_t)(sign | 0x7C00u | (man ? 0x200u : 0));
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        int shift = 14 - exp;
        uint32_t half = 1u << (shift - 1), val = man >> shift;
        if ((man & ((1u << shift) - 1)) > half ||
            ((man & ((1u << shift) - 1)) == half && (val & 1))) val++;
        return (uint16_t)(sign | val);
    }
    uint32_t half = 0x1000u;
    if ((man & 0x1FFFu) > half || ((man & 0x1FFFu) == half && (man & 0x2000u))) man += 0x2000u;
    if (man & 0x800000u) { man = 0; exp++; if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u); }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
}

static f32 f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, u;
    if (exp == 0) {
        if (man == 0) u = sign;
        else {
            int e = -1; uint32_t m = man;
            do { m <<= 1; e++; } while (!(m & 0x400u));
            m &= 0x3FFu;
            u = sign | ((uint32_t)(127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) u = sign | 0x7F800000u | (man << 13);
    else u = sign | ((exp - 15 + 127) << 23) | (man << 13);
    f32 r; memcpy(&r, &u, 4); return r;
}

// ------------------------------------------------------------------ 规格
struct HWSpec {
    double freq_hz = 1e9, dram_bw = 51.2e9, pe_util = 0.62, dram_util = 0.85;
    double freq_mhz_ = 1000.0;
    double vec_lanes = 128;      // 向量单元 lane 数 (Norm/激活/RoPE 等逐元素算子)
    long long mac_per_cycle = 1024;
    size_t sram_bytes = 512 * 1024, sram_total = 1024 * 1024;
    double pcie_bw = 3.94e9, pcie_util = 0.75;
    std::unordered_map<string, double> overhead;   // 每算子固定开销 (cycle)
    double ovh(const string &op) const {
        auto it = overhead.find(op);
        return it == overhead.end() ? 0.0 : it->second;
    }
};

struct OpStat {
    string kind, tag, phase = "init";
    long long macs = 0, dram_bytes = 0, sram_bytes = 0, elements = 0;
    double cyc_compute = 0, cyc_mem = 0, cyc = 0;
    double ms = 0;
};

// ------------------------------------------------------------------ 张量
struct Tensor {
    string name, dtype;
    bool image_backed = false;
    vector<long long> dims;          // 逻辑形状
    const uint8_t *base = nullptr;   // 镜像段指针 / arena 指针
    size_t bytes = 0;
    // 权重专有
    int group = 128, ngroup = 0, bits = 16;
    bool packed4 = false, symmetric = false;
    bool zp_int16 = false;
    vector<f32> scale, zp, inv;      // 已解码到内存, 供快速索引
    long long N = 0, K = 0, Kp = 0;

    long long elems() const {
        long long e = 1;
        for (auto d : dims) e *= d;
        return e;
    }
    int elem_bytes() const { return dtype == "fp32" ? 4 : (dtype == "fp16" ? 2 : 1); }
};

static inline f32 rd(const Tensor &t, size_t idx) {
    if (t.dtype == "fp32") return ((const f32 *)t.base)[idx];
    if (t.dtype == "fp16") return f16_to_f32(((const uint16_t *)t.base)[idx]);
    return (f32)t.base[idx];
}

static inline void wr(const Tensor &t, size_t idx, f32 v) {
    if (t.dtype == "fp32") ((f32 *)t.base)[idx] = v;
    else if (t.dtype == "fp16") ((uint16_t *)t.base)[idx] = f32_to_f16(v);
    else ((int8_t *)t.base)[idx] = (int8_t)std::max(-128.f, std::min(127.f, std::round(v)));
}

// 权重元素 (含 int4 解包 + 分组反量化 + AWQ 反缩放)
static inline f32 wgt(const Tensor &w, long long n, long long k, bool use_inv = true) {
    long long idx = n * w.Kp + k;
    f32 q;
    if (w.packed4) {
        const uint8_t *row = w.base + (size_t)(n * (w.Kp / 2));
        uint8_t b = row[k / 2];
        q = (k % 2 == 0) ? (f32)(b & 0x0F) : (f32)((b >> 4) & 0x0F);
    } else {
        // int8 权重按无符号读 (q ∈ [0,255], 再由 (q - zp) * scale 去量化)
        q = (w.dtype == "fp16") ? rd(w, (size_t)idx) : (f32)((const uint8_t *)w.base)[idx];
    }
    if (w.dtype == "fp16") return q;
    int g = (int)(k / w.group);
    f32 v = (q - w.zp[(size_t)n * w.ngroup + g]) * w.scale[(size_t)n * w.ngroup + g];
    if (use_inv && !w.inv.empty()) v *= w.inv[(size_t)k];
    return v;
}

// --------------------------------------------------------------- 设备镜像
// 设备侧只读一个 flat idx 侧车 (由 s3_pack.py 生成), 不在设备上塞 JSON 解析器。
struct Device {
    string bin_path;
    vector<uint8_t> blob;                     // 整个镜像 (mmap 亦可, 这里直接读)
    size_t body_off = 0;                      // manifest 里的 tensor 偏移是相对 body 的
    std::unordered_map<string, Tensor> tensors;
    int hidden = 0, layers = 0, heads = 0, kv_heads = 0, head_dim = 0, inter = 0, vocab = 0;
    f32 rms_eps = 1e-6f, rope_theta = 1e6f, attn_scale = 0;
    size_t used_bytes = 0;

    const Tensor &T(const string &n) const {
        auto it = tensors.find(n);
        if (it == tensors.end()) die("未声明的张量: " + n);
        return it->second;
    }
    bool has(const string &n) const { return tensors.count(n) > 0; }

    void load(const string &bin, const string &idx) {
        bin_path = bin;
        std::ifstream f(bin, std::ios::binary);
        if (!f) die("打不开镜像 " + bin);
        blob.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (blob.size() < 16 || memcmp(blob.data(), "MTXNPU01", 8) != 0)
            die("设备镜像 magic 不对 (期望 MTXNPU01): " + bin);
        std::ifstream fi(idx);
        if (!fi) die("打不开 idx " + idx);
        string line;
        while (std::getline(fi, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            string kind;
            ss >> kind;
            if (kind == "config") {
                string k, eq; double v; ss >> k >> eq >> v;
                if (k == "hidden") hidden = (int)v;
                else if (k == "body_off") body_off = (size_t)v;
                else if (k == "layers") layers = (int)v;
                else if (k == "heads") heads = (int)v;
                else if (k == "kv_heads") kv_heads = (int)v;
                else if (k == "head_dim") head_dim = (int)v;
                else if (k == "inter") inter = (int)v;
                else if (k == "vocab") vocab = (int)v;
                else if (k == "rms_eps") rms_eps = (f32)v;
                else if (k == "rope_theta") rope_theta = (f32)v;
            } else if (kind == "tensor") {
                tensor_from_idx(line);
            }
        }
        attn_scale = (f32)(1.0 / sqrt((double)head_dim));
        used_bytes = blob.size();
    }

    void tensor_from_idx(const string &line) {
        std::istringstream ss(line);
        string kind, name, role, dtype, kv;
        ss >> kind >> name >> role >> dtype;
        Tensor t; t.name = name; t.dtype = dtype;
        std::map<string, string> m;
        while (ss >> kv) {
            auto p = kv.find('=');
            if (p != string::npos) m[kv.substr(0, p)] = kv.substr(p + 1);
        }
        auto num = [&](const string &k, long long d = 0) -> long long {
            auto it = m.find(k); return it == m.end() ? d : atoll(it->second.c_str());
        };
        if (role == "weight") {
            t.image_backed = true;
            t.N = num("N"); t.K = num("K");
            if (dtype == "fp16") {
                t.dims = {t.N, t.K};
                t.bytes = (size_t)num("bytes");
                t.base = blob.data() + body_off + num("off");
                t.bits = 16;
                t.Kp = t.K;
            } else {
                t.bits = atoi(dtype.c_str() + 3);
                t.group = (int)num("group", 128);
                t.ngroup = (int)num("ngroup");
                t.packed4 = num("packed") != 0;
                t.symmetric = num("sym") != 0;
                t.zp_int16 = (m["zp_dtype"] == "int16");
                t.Kp = t.packed4 ? (long long)t.ngroup * t.group : t.ngroup * t.group;
                t.dims = {t.N, t.Kp};
                t.base = blob.data() + body_off + num("q_off");
                t.bytes = (size_t)(num("q_bytes") + num("scale_bytes") + num("zp_bytes") +
                                   num("inv_bytes"));
                const uint16_t *sc = (const uint16_t *)(blob.data() + body_off + num("scale_off"));
                const uint8_t *zpb = blob.data() + body_off + num("zp_off");
                t.scale.resize((size_t)t.N * t.ngroup);
                t.zp.resize((size_t)t.N * t.ngroup);
                for (size_t i = 0; i < t.scale.size(); i++) {
                    t.scale[i] = f16_to_f32(sc[i]);
                    t.zp[i] = t.zp_int16 ? (f32)((const int16_t *)zpb)[i] : (f32)((const int8_t *)zpb)[i];
                }
                if (num("inv_bytes") > 0) {
                    const uint16_t *iv = (const uint16_t *)(blob.data() + body_off + num("inv_off"));
                    t.inv.resize(t.K);
                    for (long long i = 0; i < t.K; i++) t.inv[i] = f16_to_f32(iv[i]);
                }
            }
        }
        tensors[name] = t;
    }

    // 权重格式自检: 把每个权重张量按硬件口径反量化后 dump 出来
    void dump_dequant(const string &dir) const {
        for (auto &kv : tensors) {
            const Tensor &t = kv.second;
            if (!t.image_backed) continue;
            string p = dir + "/" + t.name + ".f32";
            FILE *f = fopen(p.c_str(), "wb");
            if (!f) die("写不开 " + p);
            vector<f32> row(t.K);
            for (long long n = 0; n < t.N; n++) {
                for (long long k = 0; k < t.K; k++) row[k] = wgt(t, n, k, false);
                fwrite(row.data(), sizeof(f32), t.K, f);
            }
            fclose(f);
        }
    }
};

// ------------------------------------------------------------------ 执行器
struct Runner {
    const HWSpec &hw;
    Device &dev;
    string out_dir;
    std::unordered_map<string, vector<uint8_t>> store;   // 激活/KV 的 DRAM 存储
    std::unordered_map<string, Tensor> act;
    std::unordered_map<string, vector<f32>> kv_scale;    // int8 KV 的逐 (head, token) scale
    std::unordered_map<string, long long> kv_filled;
    vector<OpStat> stats;
    vector<f32> tokens;
    long long T = 1;
    string phase = "init";
    double cycles = 0, dram_bytes = 0, pcie_bytes = 0, sram_peak = 0;
    long long macs_total = 0, kv_bytes_peak = 0;
    bool dump_layers = false;
    double dram_bytes_per_cycle() const { return hw.dram_bw / hw.freq_hz; }

    Runner(const HWSpec &h, Device &d, const string &od) : hw(h), dev(d), out_dir(od) {}

    Tensor &mk(const string &name, vector<long long> dims, const string &dtype) {
        Tensor t; t.name = name; t.dtype = dtype; t.dims = dims;
        t.bytes = (size_t)t.elems() * t.elem_bytes();
        auto &buf = store[name];
        buf.assign(t.bytes, 0);
        t.base = buf.data();
        act[name] = t;
        return act[name];
    }
    Tensor &A(const string &n) {
        auto it = act.find(n);
        if (it == act.end()) die("未声明的激活张量: " + n);
        return it->second;
    }
    void kv_declare(const string &name, long long maxctx, const string &dtype) {
        Tensor t; t.name = name; t.dtype = dtype;
        t.dims = {2, dev.kv_heads, maxctx, dev.head_dim};
        t.bytes = (size_t)t.elems() * t.elem_bytes();
        auto &buf = store[name];
        buf.assign(t.bytes, 0);
        t.base = buf.data();
        act[name] = t;
        kv_filled[name] = 0;
        if (dtype == "int8") kv_scale[name].assign((size_t)2 * dev.kv_heads * maxctx, 1.0f);
        kv_bytes_peak = std::max(kv_bytes_peak, (long long)t.bytes);
    }

    void rec(const string &kind, const string &tag, long long macs, long long dram,
             long long sram, long long elems, long long vec_elems = 0) {
        OpStat s;
        s.kind = kind; s.tag = tag; s.phase = phase; s.macs = macs; s.dram_bytes = dram;
        s.sram_bytes = sram; s.elements = elems;
        double cc = (double)macs / (hw.mac_per_cycle * hw.pe_util);
        double vc = (double)vec_elems / (hw.vec_lanes * hw.pe_util);
        s.cyc_compute = std::max(cc, vc);
        s.cyc_mem = (double)dram / (dram_bytes_per_cycle() * hw.dram_util);
        s.cyc = std::max(s.cyc_compute, s.cyc_mem) + hw.ovh(kind);
        s.ms = s.cyc / hw.freq_hz * 1e3;
        stats.push_back(s);
        cycles += s.cyc;
        dram_bytes += dram;
        macs_total += macs;
        sram_peak = std::max(sram_peak, (double)sram);
    }

    // ------------------------------------------------------------ 指令实现
    void op_embed(const string &out, const string &table, long long tok_off, long long roff = 0) {
        const Tensor &E = dev.T(table);
        Tensor &O = A(out);
        long long H = dev.hidden;
        for (long long t = 0; t < T; t++) {
            long long ti = tok_off + t;
            if (ti < 0 || ti >= (long long)tokens.size()) die("token 索引越界: " + std::to_string(ti));
            long long id = (long long)tokens[ti];
            for (long long h = 0; h < H; h++) wr(O, (roff + t) * H + h, wgt(E, id, h));
        }
        long long wb = (long long)(T * H * (E.bits == 4 ? 0.5 : (E.bits == 8 ? 1 : 2)));
        rec("EMBED", out, 0, wb + T * H * 2 + T * H * 4, 0, T * H);
    }

    void op_gemm(const string &out, const string &a, const string &wname, const string &bias,
                 long long ntile, long long ktile, const string &tag) {
        const Tensor &W = dev.T(wname);
        const Tensor &A_ = A(a);
        Tensor &O = A(out);
        long long N = W.N, K = W.K;
        if (ktile <= 0 || ktile > K) ktile = K;
        // 硬件数据流: 权重一行 (一个 N_tile 的行) 解包+去量化到片上缓冲, 然后对 T 个
        // 激活做 MAC。这里按同样的顺序 (外层 N, 内层 T) 实现, 解量化每行只做一次。
        const f32 *bias_ptr = nullptr;
        if (!bias.empty()) bias_ptr = (const f32 *)dev.T(bias).base;   // 目前 bias 都是 fp16
        vector<f32> bias_v;
        if (!bias.empty()) {
            bias_v.resize(N);
            for (long long n = 0; n < N; n++) bias_v[n] = rd(dev.T(bias), n);
        }
        (void)bias_ptr;
        vector<f32> af((size_t)T * K);            // 激活 tile -> fp32 (硬件里是 SRAM 里的 fp16 tile)
#pragma omp parallel for schedule(static)
        for (long long i = 0; i < T * K; i++) af[i] = rd(A_, i);
#pragma omp parallel
        {
            vector<f32> wrow(K);
#pragma omp for schedule(static)
            for (long long n = 0; n < N; n++) {
                for (long long k = 0; k < K; k++) wrow[k] = wgt(W, n, k);
                for (long long t = 0; t < T; t++) {
                    const f32 *arow = &af[(size_t)t * K];
                    f32 acc = 0;
                    for (long long k = 0; k < K; k++) acc += arow[k] * wrow[k];
                    if (!bias_v.empty()) acc += bias_v[n];
                    wr(O, t * N + n, acc);
                }
            }
        }
        long long nt = (N + ntile - 1) / ntile, kt = (K + ktile - 1) / ktile;
        long long wb = (long long)(N * W.Kp * (W.bits == 4 ? 0.5 : (W.bits == 8 ? 1 : 2)));
        wb += (long long)(N * W.ngroup * 3);                       // scale fp16 + zp int8
        long long ab = (long long)(T * K * 2) * kt + (long long)(T * N * 4) * (kt - 1);
        long long ob = (long long)(T * N * 2);
        long long sram = 2 * (long long)(ntile * ktile * 0.5 + T * ktile * 2 + T * ntile * 4);
        rec(W.bits == 4 ? "GEMM4" : (W.bits == 8 ? "GEMM8" : "GEMMF"), tag + "/nt=" + std::to_string(nt),
            (long long)T * N * K, wb + ab + ob, sram, (long long)T * N);
    }

    void op_rmsnorm(const string &out, const string &a, const string &w) {
        const Tensor &W = dev.T(w);
        const Tensor &X = A(a);
        Tensor &O = A(out);
        long long H = dev.hidden;
        for (long long t = 0; t < T; t++) {
            double s = 0;
            for (long long h = 0; h < H; h++) { double v = rd(X, t * H + h); s += v * v; }
            f32 inv = (f32)(1.0 / sqrt(s / H + dev.rms_eps));
            for (long long h = 0; h < H; h++)
                wr(O, t * H + h, (f32)(rd(X, t * H + h) * inv) * rd(W, h));
        }
        rec("RMSNORM", out, 2 * (long long)T * H, T * H * 4 + H * 2, 0, T * H, T * H);
    }

    void op_layernorm(const string &out, const string &a, const string &w, const string &b) {
        const Tensor &W = dev.T(w), &B = dev.T(b);
        const Tensor &X = A(a);
        Tensor &O = A(out);
        long long H = dev.hidden;
        for (long long t = 0; t < T; t++) {
            double s = 0;
            for (long long h = 0; h < H; h++) s += rd(X, t * H + h);
            double m = s / H, v = 0;
            for (long long h = 0; h < H; h++) { double d = rd(X, t * H + h) - m; v += d * d; }
            f32 inv = (f32)(1.0 / sqrt(v / H + 1e-12));
            for (long long h = 0; h < H; h++)
                wr(O, t * H + h, (f32)((rd(X, t * H + h) - m) * inv) * rd(W, h) + rd(B, h));
        }
        rec("LAYERNORM", out, 3 * (long long)T * H, T * H * 4 + H * 4, 0, T * H, T * H);
    }

    void op_rope(const string &qn, const string &kn, long long pos) {
        Tensor &Q = A(qn);
        Tensor &K = A(kn);
        long long D = dev.head_dim, half = D / 2;
        for (long long t = 0; t < T; t++) {
            for (long long d = 0; d < half; d++) {
                double ang = (double)(pos + t) / pow((double)dev.rope_theta, (double)d / half);
                f32 c = f16_to_f32(f32_to_f16((f32)cos(ang)));
                f32 s = f16_to_f32(f32_to_f16((f32)sin(ang)));
                auto rot = [&](Tensor &X, long long h, long long base) {
                    f32 x1 = rd(X, base + d), x2 = rd(X, base + d + half);
                    wr(X, base + d, x1 * c - x2 * s);
                    wr(X, base + d + half, x2 * c + x1 * s);
                };
                for (long long h = 0; h < dev.heads; h++) rot(Q, h, (t * dev.heads + h) * D);
                for (long long h = 0; h < dev.kv_heads; h++) rot(K, h, (t * dev.kv_heads + h) * D);
            }
        }
        long long n = T * (dev.heads + dev.kv_heads) * D;
        rec("ROPE", qn, 3 * n, n * 4, 0, n, n);
    }

    void op_kvwrite(const string &kcn, const string &vcn, const string &kn, const string &vn,
                    long long pos, bool int8kv) {
        Tensor &KC = A(kcn);
        Tensor &VC = A(vcn);
        const Tensor &K = A(kn);
        const Tensor &V = A(vn);
        long long D = dev.head_dim, kh = dev.kv_heads;
        auto &ks = kv_scale[kcn];
        auto &vs = kv_scale[vcn];
        for (long long t = 0; t < T; t++) {
            long long p = pos + t;
            for (long long h = 0; h < kh; h++) {
                if (int8kv) {
                    double amax = 0, bmax = 0;
                    for (long long d = 0; d < D; d++) {
                        amax = std::max(amax, std::fabs((double)rd(K, (t * kh + h) * D + d)));
                        bmax = std::max(bmax, std::fabs((double)rd(V, (t * kh + h) * D + d)));
                    }
                    f32 sa = (f32)(amax / 127.0), sb = (f32)(bmax / 127.0);
                    ks[(size_t)h * KC.dims[2] + p] = sa;
                    vs[(size_t)h * VC.dims[2] + p] = sb;
                    for (long long d = 0; d < D; d++) {
                        wr(KC, ((1 * kh + h) * KC.dims[2] + p) * D + d, rd(K, (t * kh + h) * D + d) / (sa == 0 ? 1 : sa));
                        wr(VC, ((1 * kh + h) * VC.dims[2] + p) * D + d, rd(V, (t * kh + h) * D + d) / (sb == 0 ? 1 : sb));
                    }
                } else {
                    for (long long d = 0; d < D; d++) {
                        wr(KC, ((1 * kh + h) * KC.dims[2] + p) * D + d, rd(K, (t * kh + h) * D + d));
                        wr(VC, ((1 * kh + h) * VC.dims[2] + p) * D + d, rd(V, (t * kh + h) * D + d));
                    }
                }
            }
        }
        kv_filled[kcn] = std::max(kv_filled[kcn], pos + T);
        long long bytes = 2 * (long long)T * kh * D * (int8kv ? 1 : 2);
        rec("KVAPPEND", kcn, 0, bytes * 2 + (int8kv ? T * kh * 8 : 0), 0, 2 * T * kh * D);
    }

    f32 kv_read(const Tensor &C, const string &cname, long long lvl, long long h, long long s,
                long long d) {
        size_t idx = (( (size_t)lvl * C.dims[1] + h) * C.dims[2] + s) * C.dims[3] + d;
        f32 v = rd(C, idx);
        if (C.dtype == "int8") {
            const auto &sc = kv_scale.at(cname);
            v *= sc[(size_t)h * C.dims[2] + s];
        }
        return v;
    }

    void op_attn(const string &out, const string &qn, const string &kcn, const string &vcn,
                 long long pos, bool causal) {
        const Tensor &Q = A(qn);
        const Tensor &KC = A(kcn);
        const Tensor &VC = A(vcn);
        Tensor &O = A(out);
        long long D = dev.head_dim, Hq = dev.heads, kh = dev.kv_heads;
        long long S = kv_filled[kcn];
        long long rep = Hq / kh;
        // KV Cache 载入到片上缓冲 (fp16 -> fp32), int8 KV 在这里做反量化
        vector<f32> kbuf((size_t)kh * S * D), vbuf((size_t)kh * S * D);
#pragma omp parallel for schedule(static)
        for (long long i = 0; i < (long long)kh * S * D; i++) {
            long long h = i / (S * D), rest = i % (S * D), s = rest / D, d = rest % D;
            kbuf[i] = kv_read(KC, kcn, 1, h, s, d);
            vbuf[i] = kv_read(VC, vcn, 1, h, s, d);
        }
        vector<f32> qf((size_t)T * Hq * D);
        for (long long i = 0; i < T * Hq * D; i++) qf[i] = rd(Q, i);
        vector<f32> outf((size_t)T * Hq * D, 0.f);
#pragma omp parallel for schedule(static)
        for (long long th = 0; th < T * Hq; th++) {
            long long t = th / Hq, h = th % Hq;
            long long hkv = h / rep;
            vector<f32> sc(S), p(S);
            const f32 *qh = &qf[th * D];
            const f32 *kb = &kbuf[(size_t)hkv * S * D];
            double mx = -1e30;
            for (long long s = 0; s < S; s++) {
                if (causal && s > pos + t) { sc[s] = -1e30f; continue; }
                const f32 *ks = &kb[s * D];
                f32 acc = 0;
                for (long long d = 0; d < D; d++) acc += qh[d] * ks[d];
                sc[s] = f16_to_f32(f32_to_f16(acc * dev.attn_scale));
                mx = std::max(mx, (double)sc[s]);
            }
            double sum = 0;
            for (long long s = 0; s < S; s++) { p[s] = (f32)exp((double)sc[s] - mx); sum += p[s]; }
            for (long long s = 0; s < S; s++) p[s] = f16_to_f32(f32_to_f16((f32)(p[s] / sum)));
            f32 *oh = &outf[th * D];
            const f32 *vb = &vbuf[(size_t)hkv * S * D];
            for (long long s = 0; s < S; s++) {
                if (p[s] == 0.f) continue;
                const f32 *vs = &vb[s * D];
                for (long long d = 0; d < D; d++) oh[d] += p[s] * vs[d];
            }
        }
        for (long long i = 0; i < T * Hq * D; i++) wr(O, i, outf[i]);
        long long macs = 2 * Hq * T * S * D;
        long long kvread = 2 * kh * S * D * (KC.dtype == "int8" ? 1 : 2);
        long long sram = (long long)Hq * S * 4 * 2;
        // KV 读一次 (all queries 共享), Q 读 + O 写各一次
        rec("ATTNGQA", (causal ? "prefill" : "decode") + string("/S=") + std::to_string(S),
            macs, kvread + (long long)T * Hq * D * 4, sram, (long long)T * Hq * D);
    }

    void op_silumul(const string &out, const string &g, const std::string &u) {
        const Tensor &G = A(g);
        const Tensor &U = A(u);
        Tensor &O = A(out);
        long long n = (long long)T * dev.inter;
        for (long long i = 0; i < n; i++) {
            f32 gv = rd(G, i);
            f32 sv = f16_to_f32(f32_to_f16((f32)(gv / (1.0 + exp(-(double)gv)))));
            wr(O, i, sv * rd(U, i));
        }
        rec("SILUMUL", out, 6 * n, n * 6, 0, n, n);
    }

    void op_gelu(const string &out, const string &a) {
        const Tensor &X = A(a);
        Tensor &O = A(out);
        long long n = X.elems() / X.elem_bytes() * X.elem_bytes();  // 元素数
        n = (long long)X.elems();
        for (long long i = 0; i < n; i++) {
            f32 x = rd(X, i);
            wr(O, i, 0.5f * x * (1.f + erff((f32)(x / sqrt(2.0)))));
        }
        rec("GELU", out, 8 * n, n * 4, 0, n, n);
    }

    void op_add(const string &out, const string &a, const string &b) {
        const Tensor &X = A(a);
        const Tensor &Y = A(b);
        Tensor &O = A(out);
        long long n = (long long)X.elems();
        for (long long i = 0; i < n; i++) wr(O, i, rd(X, i) + rd(Y, i));
        rec("ADD", out, n, n * 6, 0, n, n);
    }

    void op_vscale(const string &out, const string &a, const string &wname) {
        const Tensor &S = dev.T(wname);
        const Tensor &X = A(a);
        Tensor &O = A(out);
        long long H = S.K;
        long long rows = (long long)X.elems() / H;
        for (long long t = 0; t < rows; t++)
            for (long long h = 0; h < H; h++) {
                f32 s = 1.0f;
                if (!S.inv.empty()) s = S.inv[h];
                else s = f16_to_f32(((const uint16_t *)S.base)[h]);
                wr(O, t * H + h, rd(X, t * H + h) * s);
            }
        rec("VSCALE", out, rows * H, rows * H * 4 + H * 2, 0, rows * H, rows * H);
    }

    void op_lmhead(const string &out, const string &a, const string &wname) {
        const Tensor &E = dev.T(wname);
        const Tensor &X = A(a);
        Tensor &O = A(out);
        long long H = dev.hidden, V = E.N;
        vector<f32> af((size_t)T * H);
        for (long long i = 0; i < T * H; i++) af[i] = rd(X, i);
#pragma omp parallel
        {
            vector<f32> wrow(H);
#pragma omp for schedule(static)
            for (long long v = 0; v < V; v++) {
                for (long long h = 0; h < H; h++) wrow[h] = wgt(E, v, h);
                for (long long t = 0; t < T; t++) {
                    const f32 *arow = &af[(size_t)t * H];
                    f32 acc = 0;
                    for (long long h = 0; h < H; h++) acc += arow[h] * wrow[h];
                    wr(O, t * V + v, acc);
                }
            }
        }
        long long wb = (long long)(V * E.Kp * (E.bits == 4 ? 0.5 : (E.bits == 8 ? 1 : 2)));
        rec("LMHEAD", out, (long long)T * V * H, wb + T * H * 2 + T * V * 4, 0, (long long)T * V);
    }

    void op_readback(const string &file, const string &buf) {
        const Tensor &X = A(buf);
        long long rows = std::min((long long)X.dims[0], T);      // 只回传当前有效行 (T)
        long long n = rows * ((long long)X.elems() / X.dims[0]);
        vector<f32> host(n);
        for (long long i = 0; i < n; i++) host[i] = rd(X, i);
        FILE *f = fopen(file.c_str(), "wb");
        if (!f) die("写不开 " + file);
        fwrite(host.data(), sizeof(f32), n, f);
        fclose(f);
        double cyc = (double)(n * 4) / (hw.pcie_bw * hw.pcie_util / hw.freq_hz);
        OpStat s; s.kind = "READBACK"; s.tag = buf; s.dram_bytes = n * 4;
        s.cyc = cyc + hw.ovh("DMA"); s.ms = s.cyc / hw.freq_hz * 1e3;
        stats.push_back(s);
        cycles += s.cyc;
        pcie_bytes += n * 4;
    }

    // 采样器跑在 Host CPU 上 (设备只回传 logits), 贪心 argmax 后把 token 追加进 token 流
    void op_sample(const string &buf, long long row = -1) {
        const Tensor &X = A(buf);
        long long V = X.dims[1];
        long long r = (row < 0) ? (T - 1) : row;
        long long best = 0;
        f32 bv = -1e30f;
        for (long long v = 0; v < V; v++) {
            f32 val = rd(X, r * V + v);
            if (val > bv) { bv = val; best = v; }
        }
        tokens.push_back((f32)best);
        rec("SAMPLE", buf, 0, V * 4 + 4, 0, V);
    }

    void op_host2dev(const string &what, long long bytes) {
        double cyc = (double)bytes / (hw.pcie_bw * hw.pcie_util / hw.freq_hz) + hw.ovh("DMA");
        OpStat s; s.kind = "HOST2DEV"; s.tag = what; s.dram_bytes = bytes;
        s.cyc = cyc; s.ms = cyc / hw.freq_hz * 1e3;
        stats.push_back(s);
        cycles += cyc;
        pcie_bytes += bytes;
    }

    // Host -> Device 的激活/输入搬运 (图像、embedding、token 表等)
    void op_load_host(const string &file, const string &buf, long long roff) {
        Tensor &O = A(buf);
        FILE *f = fopen(file.c_str(), "rb");
        if (!f) die("打不开 host 张量 " + file);
        fseek(f, 0, SEEK_END);
        long nbytes = ftell(f);
        fseek(f, 0, SEEK_SET);
        vector<f32> host(nbytes / 4);
        if (nbytes && fread(host.data(), 4, host.size(), f) != host.size()) die("读短 " + file);
        fclose(f);
        long long cols = (long long)O.elems() / O.dims[0];
        long long rows = (long long)host.size() / cols;
        if ((long long)host.size() % cols != 0) die("host 张量行宽与设备缓冲不匹配: " + file);
        for (long long r = 0; r < rows; r++)
            for (long long c = 0; c < cols; c++)
                wr(O, (roff + r) * cols + c, host[r * cols + c]);
        double cyc = (double)nbytes / (hw.pcie_bw * hw.pcie_util / hw.freq_hz);
        OpStat s;
        s.kind = "HOST2DEV"; s.tag = buf; s.phase = phase;
        s.dram_bytes = nbytes + (long long)host.size() * 2;
        s.cyc = cyc + hw.ovh("DMA");
        s.ms = s.cyc / hw.freq_hz * 1e3;
        stats.push_back(s);
        cycles += s.cyc;
        pcie_bytes += nbytes;
    }

    // 图像 im2col: [3,224,224] fp16 -> [P, 3*16*16] fp16 (ViT patch 展开)
    void op_im2col(const string &imgn, const string &out, long long patch, long long img_size,
                   long long ch) {
        const Tensor &I = A(imgn);
        Tensor &O = A(out);
        long long grid = img_size / patch;
        long long p = 0;
        for (long long gy = 0; gy < grid; gy++)
            for (long long gx = 0; gx < grid; gx++, p++) {
                long long k = 0;
                for (long long c = 0; c < ch; c++)
                    for (long long ry = 0; ry < patch; ry++)
                        for (long long rx = 0; rx < patch; rx++, k++) {
                            long long sy = gy * patch + ry, sx = gx * patch + rx;
                            wr(O, p * ch * patch * patch + k,
                               rd(I, (c * img_size + sy) * img_size + sx));
                        }
            }
        long long bytes = (long long)grid * grid * ch * patch * patch;
        rec("IM2COL", out, 0, (long long)ch * img_size * img_size * 2 + bytes * 2, 0, bytes);
    }

    // 逐行加表: out[i] = (i < rows(a) ? a[i] : 0) + tab[i]  (ViT 的 cls token / pos embed)
    void op_addtab(const string &out, const string &a, const string &tabn) {
        const Tensor &A_ = A(a);
        const Tensor &Tb = A(tabn);
        Tensor &O = A(out);
        long long cols = (long long)Tb.elems() / Tb.dims[0];
        long long ra = (long long)A_.elems() / ((long long)A_.elems() / A_.dims[0]);
        for (long long r = 0; r < (long long)Tb.dims[0]; r++)
            for (long long c = 0; c < cols; c++)
                wr(O, r * cols + c,
                   (r < ra ? rd(A_, r * cols + c) : 0.f) + rd(Tb, r * cols + c));
        long long n = (long long)Tb.elems();
        rec("ADDTAB", out, 0, n * 6 + (long long)A_.elems() * 2, 0, n, n);
    }

    // ------------------------------------------------------------ 微码解析
    void run_program(const string &path) {
        std::ifstream f(path);
        if (!f) die("打不开微码 " + path);
        string line;
        long long lineno = 0;
        while (std::getline(f, line)) {
            lineno++;
            auto h = line.find('#');
            if (h != string::npos) line = line.substr(0, h);
            std::istringstream ss(line);
            string op;
            if (!(ss >> op)) continue;
            std::map<string, string> kv;
            string tok;
            while (ss >> tok) {
                auto p = tok.find('=');
                if (p != string::npos) kv[tok.substr(0, p)] = tok.substr(p + 1);
            }
            auto g = [&](const string &k, const string &d = "") {
                auto it = kv.find(k); return it == kv.end() ? d : it->second;
            };
            auto gi = [&](const string &k, long long d = 0) { return atoll(g(k, std::to_string(d)).c_str()); };
            try {
                if (op == "IMAGE" || op == "SRAM" || op == "TOKENS") {
                    continue;                                    // 由 CLI 处理
                } else if (op == "ACT" || op == "ACT3") {
                    // 形如: ACT x 32 896 fp16  /  ACT3 q 32 14 64 fp16
                    std::istringstream s3(line);
                    vector<string> parts; string p3;
                    while (s3 >> p3) parts.push_back(p3);
                    vector<long long> dims;
                    for (size_t i = 2; i + 1 < parts.size(); i++) dims.push_back(atoll(parts[i].c_str()));
                    mk(parts[1], dims, parts.back());
                } else if (op == "KVC") {
                    // 形如: KVC kvk.L0 32 fp16
                    std::istringstream s4(line);
                    vector<string> parts4; string p4;
                    while (s4 >> p4) parts4.push_back(p4);
                    if (parts4.size() >= 4)
                        kv_declare(parts4[1], atoll(parts4[2].c_str()), parts4[3]);
                    else
                        kv_declare(g("out"), gi("maxctx", 2048), g("dtype", "fp16"));
                } else if (op == "PHASE") {
                    std::istringstream s5(line);
                    string p5, n5;
                    s5 >> p5 >> n5;
                    phase = n5;
                } else if (op == "SETT") {
                    std::istringstream s6(line);
                    vector<string> parts6; string p6;
                    while (s6 >> p6) parts6.push_back(p6);
                    T = parts6.size() >= 2 ? atoll(parts6[1].c_str()) : gi("n", 1);
                } else if (op == "EMBED") {
                    op_embed(g("out"), g("table"), gi("tok", 0), gi("roff", 0));
                } else if (op == "LOAD_HOST") {
                    op_load_host(g("file"), g("buf"), gi("roff", 0));
                } else if (op == "IM2COL") {
                    op_im2col(g("img"), g("out"), gi("patch", 16), gi("img_size", 224), gi("ch", 3));
                } else if (op == "ADDTAB") {
                    op_addtab(g("out"), g("a"), g("tab"));
                } else if (op == "GEMM") {
                    op_gemm(g("out"), g("a"), g("w"), g("bias"), gi("ntile", 256), gi("ktile", 0), g("w"));
                } else if (op == "RMSNORM") {
                    op_rmsnorm(g("out"), g("a"), g("w"));
                } else if (op == "LAYERNORM") {
                    op_layernorm(g("out"), g("a"), g("w"), g("b"));
                } else if (op == "ROPE") {
                    op_rope(g("q"), g("k"), gi("pos", 0));
                } else if (op == "KVWRITE") {
                    op_kvwrite(g("kc"), g("vc"), g("k"), g("v"), gi("pos", 0),
                               g("kv_dtype", "fp16") == "int8");
                } else if (op == "ATTN") {
                    op_attn(g("out"), g("q"), g("kc"), g("vc"), gi("pos", 0),
                            g("mode", "prefill") == "prefill");
                } else if (op == "SILUMUL") {
                    op_silumul(g("out"), g("gate"), g("up"));
                } else if (op == "GELU") {
                    op_gelu(g("out"), g("a"));
                } else if (op == "ADD") {
                    op_add(g("out"), g("a"), g("b"));
                } else if (op == "VSCALE") {
                    op_vscale(g("out"), g("a"), g("s"));
                } else if (op == "LMHEAD") {
                    op_lmhead(g("out"), g("a"), g("w"));
                } else if (op == "READBACK") {
                    op_readback(g("file", "out.bin"), g("buf"));
                } else if (op == "SAMPLE") {
                    op_sample(g("buf"), gi("row", -1));
                } else if (op == "HALT") {
                    break;
                } else {
                    die("微码第 " + std::to_string(lineno) + " 行: 未知指令 " + op);
                }
            } catch (const std::exception &e) {
                die("微码第 " + std::to_string(lineno) + " 行执行失败: " + e.what());
            }
        }
    }

    size_t dram_used() const {
        size_t tot = 0;
        for (auto &kv : store) tot += kv.second.size();
        return tot;
    }
};

// ------------------------------------------------------------------ 主程序
static HWSpec load_hw(const string &path) {
    HWSpec h;
    std::ifstream f(path);
    if (!f) {   // 没有 json 解析器: 用宽松的 key 扫描, 覆盖我们自己的 hw json
        fprintf(stderr, "[ISS] 警告: 打不开 hw 配置 %s, 用默认档位\n", path.c_str());
        return h;
    }
    string txt((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto geti = [&](const string &k, long long d) {
        auto p = txt.find("\"" + k + "\"");
        if (p == string::npos) return d;
        auto c = txt.find(':', p);
        return (long long)atoll(txt.c_str() + c + 1);
    };
    auto getd = [&](const string &k, double d) {
        auto p = txt.find("\"" + k + "\"");
        if (p == string::npos) return d;
        auto c = txt.find(':', p);
        return atof(txt.c_str() + c + 1);
    };
    h.mac_per_cycle = geti("mac_per_cycle", h.mac_per_cycle);
    h.freq_mhz_ = getd("freq_mhz", 1000.0);
    h.freq_hz = h.freq_mhz_ * 1e6;
    h.dram_bw = getd("bandwidth_gbps", 51.2) * 1e9;
    h.pe_util = getd("pe_util", 0.62);
    h.dram_util = getd("dram_util", 0.85);
    h.pcie_bw = getd("bandwidth_gbps", 3.94) * 1e9;
    h.pcie_util = getd("pcie_util", 0.75);
    h.sram_bytes = (size_t)geti("sram_usable_kib", 512) * 1024;
    h.sram_total = (size_t)geti("sram_kib", 1024) * 1024;
    h.vec_lanes = getd("vec_lanes", 128);
    // 每算子固定开销: 直接扫 op_overhead_cycles 段
    auto p = txt.find("op_overhead_cycles");
    if (p != string::npos) {
        auto e = txt.find('}', p);
        string seg = txt.substr(p, e - p);
        for (const string &k : {"GEMM4", "GEMM8", "GEMMF", "RMSNORM", "LAYERNORM", "ROPE", "ATTNGQA",
                                "SOFTMAX", "SILUMUL", "GELU", "ADD", "KVAPPEND", "EMBED", "DMA", "IM2COL"}) {
            auto pp = seg.find("\"" + k + "\"");
            if (pp == string::npos) continue;
            auto c = seg.find(':', pp);
            h.overhead[k] = atof(seg.c_str() + c + 1);
        }
    }
    return h;
}

int main(int argc, char **argv) {
    string program, image, idx, tokens_file, out_json, dump_dir, hw_path, out_dir = ".";
    long long sram_kib = 0;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        auto next = [&]() { return string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--program") program = next();
        else if (a == "--image") image = next();
        else if (a == "--idx") idx = next();
        else if (a == "--tokens") tokens_file = next();
        else if (a == "--out") out_json = next();
        else if (a == "--dump-dequant") dump_dir = next();
        else if (a == "--hw") hw_path = next();
        else if (a == "--sram-kib") sram_kib = atoll(next().c_str());
        else if (a == "--out-dir") out_dir = next();
        else die("未知参数 " + a);
    }
    HWSpec hw = load_hw(hw_path);
    if (sram_kib > 0) hw.sram_bytes = (size_t)sram_kib * 1024;

    Device dev;
    dev.load(image, idx);
    if (!dump_dir.empty()) {
        string cmd_mk = "mkdir -p '" + dump_dir + "'";
        if (system(cmd_mk.c_str()) != 0) die("mkdir 失败");
        dev.dump_dequant(dump_dir);
        printf("{\"dump_dequant\":\"%s\",\"tensors\":%zu,\"bytes\":%zu}\n", dump_dir.c_str(),
               dev.tensors.size(), dev.used_bytes);
        return 0;
    }
    Runner R(hw, dev, out_dir);
    if (!tokens_file.empty()) {
        FILE *f = fopen(tokens_file.c_str(), "rb");
        if (!f) die("打不开 token 文件 " + tokens_file);
        fseek(f, 0, SEEK_END);
        long n = ftell(f) / sizeof(int32_t);
        fseek(f, 0, SEEK_SET);
        vector<int32_t> tmp((size_t)std::max(0L, n));
        if (n > 0 && fread(tmp.data(), sizeof(int32_t), n, f) != (size_t)n) die("读 token 失败");
        R.tokens.assign(tmp.begin(), tmp.end());          // int32 id -> f32 保存
        fclose(f);
    }
    auto t0 = std::chrono::steady_clock::now();
    R.run_program(program);
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // 汇总
    std::map<string, OpStat> by_kind;
    for (auto &s : R.stats) {
        auto &d = by_kind[s.kind];
        d.kind = s.kind;
        d.cyc += s.cyc; d.cyc_compute += s.cyc_compute; d.cyc_mem += s.cyc_mem;
        d.dram_bytes += s.dram_bytes; d.macs += s.macs; d.sram_bytes = std::max(d.sram_bytes, s.sram_bytes);
        d.elements += s.elements;
    }
    FILE *j = fopen(out_json.c_str(), "w");
    if (!j) die("写不开 " + out_json);
    fprintf(j, "{\n  \"ops\": [\n");
    for (size_t i = 0; i < R.stats.size(); i++) {
        auto &s = R.stats[i];
        fprintf(j, "    {\"kind\":\"%s\",\"tag\":\"%s\",\"phase\":\"%s\",\"macs\":%lld,\"dram_bytes\":%lld,"
                   "\"sram_bytes\":%lld,\"elements\":%lld,\"cycles\":%.1f,\"cycles_compute\":%.1f,"
                   "\"cycles_mem\":%.1f,\"us_at_freq\":%.3f}%s\n",
                s.kind.c_str(), s.tag.c_str(), s.phase.c_str(), s.macs, s.dram_bytes, s.sram_bytes, s.elements,
                s.cyc, s.cyc_compute, s.cyc_mem, s.cyc / hw.freq_hz * 1e6,
                i + 1 < R.stats.size() ? "," : "");
    }
    fprintf(j, "  ],\n  \"summary\": {\n");
    fprintf(j, "    \"cycles\": %.1f, \"ms\": %.4f, \"macs\": %lld, \"dram_bytes\": %.0f, "
               "\"pcie_bytes\": %.0f, \"sram_peak_bytes\": %.0f, \"dram_used_bytes\": %zu, "
               "\"image_bytes\": %zu, "
               "\"kv_peak_bytes\": %lld, \"wall_s\": %.3f, \"n_ops\": %zu,\n",
            R.cycles, R.cycles / hw.freq_hz * 1e3, R.macs_total, R.dram_bytes, R.pcie_bytes,
            R.sram_peak, R.dram_used(), dev.blob.size(), R.kv_bytes_peak, wall, R.stats.size());
    fprintf(j, "    \"by_kind\": {");
    bool first = true;
    for (auto &kv : by_kind) {
        auto &d = kv.second;
        fprintf(j, "%s\"%s\": {\"cycles\": %.1f, \"cycles_mem\": %.1f, \"dram_bytes\": %lld, "
                   "\"macs\": %lld, \"n\": %d}",
                first ? "" : ", ", kv.first.c_str(), d.cyc, d.cyc_mem, d.dram_bytes, d.macs,
                (int)std::count_if(R.stats.begin(), R.stats.end(),
                                   [&](const OpStat &s) { return s.kind == kv.first; }));
        first = false;
    }
    fprintf(j, "  }\n  }\n}\n");
    fclose(j);
    printf("{\"cycles\": %.1f, \"ms\": %.4f, \"macs\": %lld, \"dram_bytes\": %.0f, "
           "\"pcie_bytes\": %.0f, \"n_ops\": %zu, \"wall_s\": %.2f}\n",
           R.cycles, R.cycles / hw.freq_hz * 1e3, R.macs_total, R.dram_bytes, R.pcie_bytes,
           R.stats.size(), wall);
    return 0;
}
