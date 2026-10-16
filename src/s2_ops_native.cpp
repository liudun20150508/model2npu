// 原生 (C++) 参考算子实现 —— 与 src/mtx_ops_ref.py 逐算子对齐。
//
// 为什么要有这一份:
//   * 需求 要求 "C/C++ 开发能力" + "编写参考算子 + 自动化测试";
//   * NPU 侧算子是 C/C++ 写的 (s4_iss.cpp), 先用一份最小 native 实现把
//     RMSNorm/RoPE/GQA Attention/W4A16 GEMM 的数值行为钉死, 再谈上设备;
//   * 每个算子一次读 raw float32 输入、写 raw float32 输出, 由 Python 驱动
//     批量生成用例 (含边界条件) 并做判据裁决。
//
// 用法:  s2_ops_native --op rmsnorm --shape 4,8 --in x.bin --w w.bin --out y.bin --eps 1e-6
// 编译:  g++ -O3 -march=native -fopenmp -std=c++17 s2_ops_native.cpp -o build/s2_ops_native
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using std::string;
using std::vector;
using f32 = float;

// ------------------------------------------------------------------ IO
static vector<f32> read_f32(const string &p) {
    FILE *f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "打不开 %s\n", p.c_str()); exit(2); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    vector<f32> v(n / sizeof(f32));
    if (n && fread(v.data(), 1, n, f) != (size_t)n) { fprintf(stderr, "读短 %s\n", p.c_str()); exit(2); }
    fclose(f);
    return v;
}

static vector<uint8_t> read_u8(const string &p) {
    FILE *f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "打不开 %s\n", p.c_str()); exit(2); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    vector<uint8_t> v(n);
    if (n && fread(v.data(), 1, n, f) != (size_t)n) { fprintf(stderr, "读短 %s\n", p.c_str()); exit(2); }
    fclose(f);
    return v;
}

static void write_f32(const string &p, const vector<f32> &v) {
    FILE *f = fopen(p.c_str(), "wb");
    if (!f) { fprintf(stderr, "写不开 %s\n", p.c_str()); exit(2); }
    fwrite(v.data(), sizeof(f32), v.size(), f);
    fclose(f);
}

// ------------------------------------------------------------- fp16 工具
static bool g_round = true;   // 是否模拟 fp16 激活存储 (--act-fp16 控制)

// IEEE754 binary16: 1/5/10, round-to-nearest-even (与硬件 cast 单元一致)
static uint16_t f32_to_f16(f32 x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    uint32_t sign = (u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((u >> 23) & 0xFF) - 127 + 15;
    uint32_t man = u & 0x7FFFFFu;
    if (((u >> 23) & 0xFF) == 0xFF) {                       // inf / nan
        return (uint16_t)(sign | 0x7C00u | (man ? 0x200u : 0));
    }
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);     // 溢出 -> inf
    if (exp <= 0) {                                         // 次正规 / 下溢
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        int shift = 14 - exp;
        uint32_t half = 1u << (shift - 1);
        uint32_t val = man >> shift;
        if ((man & ((1u << shift) - 1)) > half || ((man & ((1u << shift) - 1)) == half && (val & 1)))
            val++;
        return (uint16_t)(sign | val);
    }
    uint32_t half = 0x1000u;                                // 尾数 round-to-nearest-even
    if ((man & 0x1FFFu) > half || ((man & 0x1FFFu) == half && (man & 0x2000u)))
        man += 0x2000u;
    if (man & 0x800000u) { man = 0; exp++; if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u); }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
}

static f32 f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t man = h & 0x3FFu;
    uint32_t u;
    if (exp == 0) {
        if (man == 0) u = sign;
        else {
            int e = -1; uint32_t m = man;
            do { m <<= 1; e++; } while (!(m & 0x400u));
            m &= 0x3FFu;
            u = sign | ((uint32_t)(127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {
        u = sign | 0x7F800000u | (man << 13);
    } else {
        u = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    f32 r; memcpy(&r, &u, 4); return r;
}

static inline f32 round_fp16(f32 x) { return f16_to_f32(f32_to_f16(x)); }
static inline f32 R(f32 x) { return g_round ? round_fp16(x) : x; }

// ---------------------------------------------------------------- 算子
static vector<f32> op_rmsnorm(const vector<f32> &x, const vector<f32> &w, int T, int H, f32 eps) {
    vector<f32> y((size_t)T * H);
#pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        const f32 *p = &x[(size_t)t * H];
        double s = 0;                              // 方差用 double 累加 -> 与 RTL 定点累加器误差同量级
        for (int h = 0; h < H; h++) s += (double)p[h] * p[h];
        f32 inv = (f32)(1.0 / sqrt(s / H + (double)eps));
        f32 *o = &y[(size_t)t * H];
        for (int h = 0; h < H; h++) o[h] = R(p[h] * inv * w[h]);
    }
    return y;
}

static vector<f32> op_layernorm(const vector<f32> &x, const vector<f32> &w, const vector<f32> &b,
                                int T, int H, f32 eps) {
    vector<f32> y((size_t)T * H);
#pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        const f32 *p = &x[(size_t)t * H];
        double s = 0;
        for (int h = 0; h < H; h++) s += p[h];
        double m = s / H, v = 0;
        for (int h = 0; h < H; h++) { double d = p[h] - m; v += d * d; }
        f32 inv = (f32)(1.0 / sqrt(v / H + (double)eps));
        f32 *o = &y[(size_t)t * H];
        for (int h = 0; h < H; h++) o[h] = R((f32)((p[h] - m) * inv) * w[h] + b[h]);
    }
    return y;
}

// x:[T,Hn,D] k 维为 head 维; theta 决定频率表
static vector<f32> op_rope(const vector<f32> &x, int T, int Hn, int D, f32 theta,
                           const vector<int> &pos) {
    vector<f32> y(x.size());
    int half = D / 2;
    vector<double> inv(half);
    for (int i = 0; i < half; i++) inv[i] = 1.0 / pow((double)theta, (double)i / half);
    // cos/sin 表按 fp16 存 (硬件是查表), 再转 fp32 参与 MAC
#pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < half; i++) {
            double ang = (double)pos[t] * inv[i];
            f32 c = R((f32)cos(ang));
            f32 s = R((f32)sin(ang));
            for (int hn = 0; hn < Hn; hn++) {
                size_t base = ((size_t)t * Hn + hn) * D;
                f32 x1 = x[base + i], x2 = x[base + i + half];
                y[base + i] = R(x1 * c - x2 * s);
                y[base + i + half] = R(x2 * c + x1 * s);
            }
        }
    }
    return y;
}

// 非因果/因果通用: q[T,Hq,D], k/v[S,Hkv,D] -> o[T,Hq,D]
static vector<f32> op_attn(const vector<f32> &q, const vector<f32> &k, const vector<f32> &v,
                           int T, int Hq, int Hkv, int S, int D, f32 scale, bool causal,
                           const vector<int> &qpos, vector<f32> *probs_out) {
    vector<f32> out((size_t)T * Hq * D);
    int rep = Hq / Hkv;
    if (probs_out) probs_out->assign((size_t)T * Hq * S, 0.f);
    for (int t = 0; t < T; t++) {
        for (int h = 0; h < Hq; h++) {
            int kh = h / rep;
            vector<f32> sc(S);
            vector<f32> p(S);
            for (int s = 0; s < S; s++) {
                double acc = 0;
                for (int d = 0; d < D; d++) {
                    f32 qv = q[((size_t)t * Hq + h) * D + d];
                    f32 kv = k[((size_t)s * Hkv + kh) * D + d];
                    acc += (double)qv * kv;
                }
                sc[s] = R((f32)(acc * scale));
            }
            for (int s = 0; s < S; s++)
                if (causal && s > qpos[t]) sc[s] = -INFINITY;
            f32 mx = -INFINITY;
            for (int s = 0; s < S; s++) mx = std::max(mx, sc[s]);
            double sum = 0;
            for (int s = 0; s < S; s++) { p[s] = (f32)exp((double)sc[s] - mx); sum += p[s]; }
            for (int s = 0; s < S; s++) p[s] = R((f32)(p[s] / sum));
            for (int s = 0; s < S; s++) {
                if (probs_out) (*probs_out)[((size_t)t * Hq + h) * S + s] = p[s];
                if (p[s] == 0.f) continue;
                for (int d = 0; d < D; d++)
                    out[((size_t)t * Hq + h) * D + d] += p[s] * v[((size_t)s * Hkv + kh) * D + d];
            }
        }
    }
    for (auto &o : out) o = R(o);
    return out;
}

static vector<f32> op_silu_mul(const vector<f32> &g, const vector<f32> &u) {
    vector<f32> y(g.size());
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < g.size(); i++) {
        f32 s = R((f32)(g[i] / (1.0 + exp(-(double)g[i]))));
        y[i] = R(s * u[i]);
    }
    return y;
}

static vector<f32> op_gelu(const vector<f32> &x) {
    vector<f32> y(x.size());
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x.size(); i++)
        y[i] = R(0.5f * x[i] * (1.f + erff((f32)(x[i] / sqrt(2.0)))));
    return y;
}

static vector<uint8_t> op_pack(const vector<uint8_t> &q, int N, int K) {
    vector<uint8_t> p((size_t)N * K / 2);
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K / 2; k++) {
            uint8_t lo = q[(size_t)n * K + 2 * k] & 0x0F;
            uint8_t hi = q[(size_t)n * K + 2 * k + 1] & 0x0F;
            p[(size_t)n * K / 2 + k] = (uint8_t)(lo | (hi << 4));
        }
    return p;
}

static void unpack_into(const uint8_t *packed, uint8_t *q, int N, int K) {
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K / 2; k++) {
            uint8_t b = packed[(size_t)n * K / 2 + k];
            q[(size_t)n * K + 2 * k] = b & 0x0F;
            q[(size_t)n * K + 2 * k + 1] = (b >> 4) & 0x0F;
        }
}

// 分组去量化 GEMM: W[N,K] 以 int4/int8 打包, scale/zp[N,G], 激活 fp16 存储, fp32 累加
static vector<f32> op_gemm_quant(const vector<f32> &x, const vector<uint8_t> &wq,
                                 const vector<f32> &scale, const vector<f32> &zp,
                                 const vector<f32> &bias,
                                 int T, int K, int N, int group, int bits, bool w_packed,
                                 bool act_fp16) {
    vector<f32> act(x.size());
    for (size_t i = 0; i < x.size(); i++) act[i] = act_fp16 ? round_fp16(x[i]) : x[i];
    vector<f32> y((size_t)T * N, 0.f);
    int G = (K + group - 1) / group;
    int Kp = G * group;
    vector<uint8_t> wfull((size_t)N * Kp, 0);
    const bool packed4 = w_packed && bits == 4;   // int4 才需要解半字节
    if (packed4) unpack_into(wq.data(), wfull.data(), N, Kp);
    else memcpy(wfull.data(), wq.data(), (size_t)N * K);
    // 每个 K 组做一次累加 (与分组数据流一致: 权重按组从 DRAM 流入, 组内 fp32 累加)
    for (int g = 0; g < G; g++) {
        int k0 = g * group, k1 = std::min(K, k0 + group);
#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; n++) {
            const f32 sc = scale[(size_t)n * G + g], zpv = zp[(size_t)n * G + g];
            const uint8_t *wp = &wfull[(size_t)n * Kp];
            for (int t = 0; t < T; t++) {
                double acc = 0;
                for (int k = k0; k < k1; k++)
                    acc += (double)act[(size_t)t * K + k] * ((f32)wp[k] - zpv) * sc;
                y[(size_t)t * N + n] += (f32)acc;
            }
        }
    }
    for (size_t i = 0; i < y.size(); i++) {
        if (!bias.empty()) y[i] = (f32)((double)y[i] + bias[i % N]);
        y[i] = R(y[i]);
    }
    (void)bits;
    return y;
}

// 激活 int8 量化 (per-token 对称) —— W8A8 的 A8 部分
// 权重反量化 (与硬件解包/去量化单元口径一致, 用于 bit 级比对)
static vector<f32> op_dequant(const vector<uint8_t> &wq, const vector<f32> &scale,
                              const vector<f32> &zp, int N, int K, int group, int bits,
                              bool w_packed) {
    int G = (K + group - 1) / group;
    int Kp = G * group;
    vector<uint8_t> wfull((size_t)N * Kp, 0);
    if (w_packed && bits == 4) unpack_into(wq.data(), wfull.data(), N, Kp);
    else memcpy(wfull.data(), wq.data(), (size_t)N * K);
    vector<f32> w((size_t)N * K);
    for (int n = 0; n < N; n++)
        for (int g = 0; g < G; g++) {
            const f32 sc = scale[(size_t)n * G + g], zpv = zp[(size_t)n * G + g];
            for (int k = g * group; k < std::min(K, (g + 1) * group); k++)
                w[(size_t)n * K + k] = ((f32)wfull[(size_t)n * Kp + k] - zpv) * sc;
        }
    return w;
}

static vector<f32> op_quant_act8(const vector<f32> &x, int T, int K) {
    vector<f32> dq(x.size());
#pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        double amax = 0;
        for (int k = 0; k < K; k++) amax = std::max(amax, fabs((double)x[(size_t)t * K + k]));
        f32 sc = (f32)(amax / 127.0);
        for (int k = 0; k < K; k++) {
            f32 q = (f32)nearbyint(x[(size_t)t * K + k] / sc);
            q = std::max(-127.f, std::min(127.f, q));
            dq[(size_t)t * K + k] = q * sc;
        }
    }
    return dq;
}

// ------------------------------------------------------------------ CLI
static vector<int> parse_ints(const char *s) {
    vector<int> v;
    const char *p = s;
    while (*p) {
        v.push_back(atoi(p));
        while (*p && *p != ',') p++;
        if (*p == ',') p++;
    }
    return v;
}

int main(int argc, char **argv) {
    string op, in1, in2, in3, wf, sf, zf, bf, out, out2, packed_f;
    vector<int> shape{1}, kvshape;
    int group = 128, bits = 4, causal = 0;
    f32 eps = 1e-6f, theta = 1e6f, scale = -1.f;
    int act_fp16 = 1, w_packed = 1, repeat = 1;
    vector<int> pos;
    auto arg = [&](int &i, string &dst) { if (i + 1 < argc) dst = argv[++i]; };
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "--op") arg(i, op);
        else if (a == "--shape") shape = parse_ints(argv[++i]);
        else if (a == "--kvshape") kvshape = parse_ints(argv[++i]);
        else if (a == "--pos") pos = parse_ints(argv[++i]);
        else if (a == "--in") arg(i, in1);
        else if (a == "--in2") arg(i, in2);
        else if (a == "--in3") arg(i, in3);
        else if (a == "--w") arg(i, wf);
        else if (a == "--scale") arg(i, sf);
        else if (a == "--zp") arg(i, zf);
        else if (a == "--bias") arg(i, bf);
        else if (a == "--packed") arg(i, packed_f);
        else if (a == "--out") arg(i, out);
        else if (a == "--out2") arg(i, out2);
        else if (a == "--group") group = atoi(argv[++i]);
        else if (a == "--bits") bits = atoi(argv[++i]);
        else if (a == "--eps") eps = (f32)atof(argv[++i]);
        else if (a == "--theta") theta = (f32)atof(argv[++i]);
        else if (a == "--attn-scale") scale = (f32)atof(argv[++i]);
        else if (a == "--causal") causal = atoi(argv[++i]);
        else if (a == "--act-fp16") { act_fp16 = atoi(argv[++i]); g_round = act_fp16 != 0; }
        else if (a == "--packed-in") w_packed = atoi(argv[++i]);
        else if (a == "--repeat") repeat = atoi(argv[++i]);
        else { fprintf(stderr, "未知参数 %s\n", a.c_str()); return 2; }
    }
    vector<f32> y;
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; r++) {
        if (op == "rmsnorm") {
            auto x = read_f32(in1); auto w = read_f32(wf);
            y = op_rmsnorm(x, w, shape[0], shape[1], eps);
        } else if (op == "layernorm") {
            auto x = read_f32(in1); auto w = read_f32(wf); auto b = read_f32(bf);
            y = op_layernorm(x, w, b, shape[0], shape[1], eps);
        } else if (op == "rope") {
            auto x = read_f32(in1);
            if (pos.empty()) for (int i = 0; i < shape[0]; i++) pos.push_back(i);
            y = op_rope(x, shape[0], shape[1], shape[2], theta, pos);
        } else if (op == "attn" || op == "attn_decode") {
            auto q = read_f32(in1); auto k = read_f32(in2); auto v = read_f32(in3);
            int T = shape[0], Hq = shape[1], D = shape[2];
            int Hkv = kvshape.size() ? kvshape[0] : Hq;
            int S = kvshape.size() > 1 ? kvshape[1] : T;
            if (pos.empty()) for (int i = 0; i < T; i++) pos.push_back(i);
            f32 sc = scale > 0 ? scale : (f32)(1.0 / sqrt((double)D));
            vector<f32> probs;
            y = op_attn(q, k, v, T, Hq, Hkv, S, D, sc,
                        op == "attn_decode" ? false : (causal != 0), pos,
                        out2.empty() ? nullptr : &probs);
            if (!out2.empty()) write_f32(out2, probs);
        } else if (op == "silu_mul") {
            auto g = read_f32(in1); auto u = read_f32(in2);
            y = op_silu_mul(g, u);
        } else if (op == "gelu") {
            auto x = read_f32(in1);
            y = op_gelu(x);
        } else if (op == "pack_int4") {
            auto q = read_u8(in1);
            y.resize(0);
            auto p = op_pack(q, shape[0], shape[1]);
            FILE *f = fopen(out.c_str(), "wb");
            fwrite(p.data(), 1, p.size(), f);
            fclose(f);
            printf("{\"op\":\"pack_int4\",\"bytes\":%zu,\"repeat\":%d}\n", p.size(), repeat);
            return 0;
        } else if (op == "gemm_w4a16" || op == "gemm_w8a16") {
            auto x = read_f32(in1);
            auto scale_v = read_f32(sf);
            auto zp_v = read_f32(zf);
            vector<f32> bias;
            if (!bf.empty()) bias = read_f32(bf);
            int T = shape[0], K = shape[1], N = shape[2];
            vector<uint8_t> wq;
            if (w_packed && bits == 4) {
                if (!packed_f.empty()) wq = read_u8(packed_f);
                else wq = op_pack(read_u8(wf), N, ((K + group - 1) / group) * group);
            } else {
                wq = read_u8(wf);
            }
            y = op_gemm_quant(x, wq, scale_v, zp_v, bias, T, K, N, group, bits, w_packed, act_fp16 != 0);
        } else if (op == "dequant") {
            auto scale_v = read_f32(sf);
            auto zp_v = read_f32(zf);
            int N = shape[0], K = shape[1];
            vector<uint8_t> wq = read_u8(w_packed && bits == 4 ? packed_f : wf);
            y = op_dequant(wq, scale_v, zp_v, N, K, group, bits, w_packed);
        } else if (op == "quant_act8") {
            auto x = read_f32(in1);
            y = op_quant_act8(x, shape[0], shape[1]);
        } else if (op == "fp16_round") {
            auto x = read_f32(in1);
            y.resize(x.size());
            for (size_t i = 0; i < x.size(); i++) y[i] = round_fp16(x[i]);
        } else {
            fprintf(stderr, "未知算子 %s\n", op.c_str());
            return 2;
        }
    }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / repeat;
    if (!out.empty()) write_f32(out, y);
    printf("{\"op\":\"%s\",\"n\":%zu,\"repeat\":%d,\"ms_per_iter\":%.3f}\n", op.c_str(), y.size(), repeat, ms);
    return 0;
}
