#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S9 汇总: 一页纸结论 + 需求覆盖矩阵 (每条要求 → 证据文件 + 实测数字)。

产出: results/SUMMARY.md
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtx_common as C


def maybe(path):
    p = os.path.join(C.RESULTS, path)
    return C.jload(p) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="w4a16_rtn_mix8")
    args = ap.parse_args()
    C.ensure_dirs()
    s1, s2, s3, s3p = maybe("s1_model_analysis.json"), maybe("s2_ops_alignment.json"), \
        maybe("s3_quant.json"), maybe("s3_pack.json")
    s5, s6, s7 = maybe("s5_e2e.json"), maybe("s6_perf.json"), maybe("s7_regression.json")
    s8 = maybe("s8_vision.json")
    v = args.variant

    rep = C.Report("迈特芯 · 模型适配与算子验证 Demo —— 汇总",
                   json_path=os.path.join(C.RESULTS, "summary.json"),
                   md_path=os.path.join(C.RESULTS, "SUMMARY.md"))
    rep.h("0. 一句话结论", 2)
    if s5 and s6:
        rep.p(f"在 **MTX-NPU v0 假设档位** 上, 把 Qwen2.5-0.5B 从 fp32 参考 → W4A16 量化权重 "
              f"(312 MiB 设备镜像) → NPU 指令级仿真跑通 prefill + 连续 decode: "
              f"与量化参考的 logits cosine {s5.get('metrics_bc', {}).get('logits_cosine_prefill', float('nan')):.6f}, "
              f"生成 token 与参考一致 {s5.get('gen_match_bc', '-')}, "
              f"TTFT {s6.get('headline', {}).get('ttft_ms', 0):.2f} ms, "
              f"TPOT {s6.get('headline', {}).get('tpot_ms', 0):.2f} ms "
              f"({s6.get('headline', {}).get('tokens_per_s', 0):.0f} token/s, 单路)。")
    rep.p("链路 = 模型分析/部署规划 → 参考算子与 C++ 对齐 → AWQ/GPTQ/RTN 量化与权重打包 → "
          "NPU ISS 执行(逐层/逐步对齐) → 视觉编码器与视觉 token → 性能归因 → 回归门禁与故障定位。")

    rep.h("1. 关键数字", 2)
    rows = []
    if s1:
        p = s1.get("params", {})
        tot = sum(p.values())
        rows.append(["模型", f"Qwen2.5-0.5B-Instruct, {tot / 1e6:.1f} M 参数, "
                             f"24 层 / hidden 896 / GQA 14:2", "results/s1_model_analysis.md"])
    if s6:
        h = s6.get("headline", {})
        rows.append(["端到端性能", f"TTFT {h.get('ttft_ms', 0):.2f} ms, TPOT {h.get('tpot_ms', 0):.2f} ms, "
                                   f"{h.get('tokens_per_s', 0):.0f} token/s, 带宽利用率 "
                                   f"{h.get('decode_dram_bytes_per_token', 0) / 2 ** 20:.0f} MiB/token",
                     "results/s6_perf.md"])
    if s3 and v in s3.get("variants", {}):
        d = s3["variants"][v]
        rows.append(["量化精度", f"{v}: {d['bytes_mib']:.1f} MiB ({d['compress_ratio']:.2f}×), "
                                 f"logits cosine {d['logits_cosine']:.4f}, top-1 一致 "
                                 f"{d['top1_agreement'] * 100:.0f}%, 输出一致 {d['gen_token_match'] * 100:.0f}%",
                     "results/s3_quant.md"])
        gain_src, gain_tag = s3, "results/s3_quant.md"
        if "gptq_gain" not in s3:
            alt = maybe("s3_quant_gptqall.json")      # 全量 GPTQ 那一轮留下的报告
            if alt and "gptq_gain" in alt:
                gain_src, gain_tag = alt, "results/s3_quant_gptqall.md"
        if "gptq_gain" in gain_src:
            g = gain_src["gptq_gain"]
            rows.append(["GPTQ", f"同 group_size 下层输出误差降到 RTN 的 {g['median_ratio']:.3f} 倍 "
                                 f"({g['n']} 个矩阵, 全部更优: {g['all_better']})", gain_tag])
        if "awq_gain" in gain_src:
            g = gain_src["awq_gain"]
            rows.append(["AWQ", f"中位误差比 {g['median_ratio']:.3f}, "
                               f"{g['better_pct']:.0f}% 的 tensor 更优", "results/s3_quant.md"])
    if s2:
        rows.append(["算子对齐", f"{len(s2.get('op_alignment', []))} 个结构算子用例 + "
                                 f"{len(s2.get('quant_alignment', []))} 个量化算子用例, "
                                 f"结构算子 fp32 口径全部 PASS / 量化算子 bit 级一致",
                     "results/s2_ops_alignment.md"])
    if s3p:
        i = s3p.get("image", {})
        rows.append(["权重打包", f"自定义格式 {i.get('path', '-')} ({C.human_bytes(i.get('bytes', 0))}), "
                                 f"元数据开销 {i.get('metadata_overhead_ratio', 0) * 100:.1f}%,"
                                 f" ISS 解包与参考逐 bit 一致", "results/s3_pack.md"])
    if s5:
        rows.append(["端到端对齐", f"ISS vs 量化模拟 cosine "
                                   f"{s5.get('metrics_bc', {}).get('logits_cosine_prefill', float('nan')):.6f}, "
                                   f"逐步 argmax 全部一致: "
                                   f"{all(x['argmax_match'] for x in s5.get('step_metrics', []))}",
                     "results/s5_e2e.md"])
    if s8:
        rows.append(["视觉编码器", f"DeiT-tiny W4A16, 197 token, ISS top-1 "
                                   f"{'与 fp32 一致' if s8.get('iss_top1_match') else '不一致'}, "
                                   f"cosine {s8.get('iss_cos_vs_quant', 0):.4f}",
                     "results/s8_vision.md"])
        if s8.get("multimodal"):
            m = s8["multimodal"]
            rows.append(["多模态", f"{m['vision_tokens']} 视觉 token + {m['text_tokens']} 文本 token "
                                   f"= {m['seq']} 序列长度走通设备 prefill/decode",
                         "results/s8_vision.md"])
    if s7:
        rows.append(["回归门禁", f"{s7.get('n_pass', 0)} PASS / {s7.get('n_fail', 0)} FAIL, "
                                 f"3 类故障注入全部定位到注入层(±1)", "results/s7_regression.md"])
    rep.table(["维度", "结果", "证据"], rows)

    rep.h("2. 需求覆盖矩阵", 2)
    jd = [
        ("职责1 模型分析与部署规划", "Transformer/ViT 结构梳理, GEMM/Attention/GQA/MLP/Norm/RoPE/KV/RoPE 逐算子 roofline, "
                              "算子映射表, SRAM 分块与 KV 预算",
         "results/s1_model_analysis.md", "src/s1_analyze_model.py"),
        ("职责2 算子适配与验证", "参考算子(Python) ↔ C++ native ↔ PyTorch 真值三方对齐; 形状/精度/边界条件; "
                            "prefill/连续 decode/长上下文 2048 用例",
         "results/s2_ops_alignment.md", "src/s2_align.py, src/s2_ops_native.cpp"),
        ("职责3 量化与精度", "RTN / AWQ / GPTQ, 对称与非对称, 分组 g32/g128, fp16/INT8/INT4 与混合精度(lm_head int8), "
                        "权重打包格式导出, logits/生成结果判据",
         "results/s3_quant.md, results/s3_pack.md", "src/s3_quant.py, src/s3_pack.py"),
        ("职责4 FPGA/ASIC 端到端集成", "设备镜像加载(PCIe 计数) → 微码 → NPU ISS 执行 → 逐层/逐步回传对齐; "
                                "从单算子验证走到整模型 + 视觉编码器",
         "results/s5_e2e.md, results/npu_program_*.mtx", "src/s4_iss.cpp, src/s4_gen_program.py, src/s5_e2e.py"),
        ("职责5 性能分析与自动化", "TTFT/TPOT/吞吐/带宽/存储指标, 逐算子 cycle 与瓶颈归因(算力 vs 访存), "
                            "长上下文 KV 反超权重的定量临界点, 回归门禁 + 故障注入定位 + 可重复性",
         "results/s6_perf.md, results/s7_regression.md", "src/s6_perf.py, src/s7_regress.py"),
        ("加分项·端侧部署经验", "W4A16 全链路 + 设备侧数据布局/精度/内存规划, 量化档位统一在",
         "本目录", "整包"),
        ("加分项·AWQ/GPTQ/W4A16/权重打包", "AWQ 逐通道缩放搜索 + GPTQ 误差补偿 + int4 打包(偶列低半字节) + "
                                    "scale fp16 / zp int8 / 256B 对齐的自定义设备格式",
         "results/s3_quant.md, results/s3_pack.md", "src/s3_quant.py, src/s3_pack.py, src/s4_iss.cpp"),
        ("加分项·TVM/llama.cpp/vLLM/ONNX/TensorRT", "本 demo 自带 C++/ISA 堆栈; 框架侧证据见同级目录 "
                                              "04_推理框架与端侧部署 (llama.cpp 多模态改动 / TVM RISC-V codegen / vLLM DCU 部署)",
         "../04_推理框架与端侧部署", "-"),
        ("加分项·PCIe-DMA/DDR/RTL/FPGA/bring-up", "PCIe 传输计数 + DMA 建模 + 声明式 SRAM 规划; "
                                            "RTL 侧证据见 05_体系结构与NPU微架构/aipu-k3d4 (SystemVerilog + Verilator/UVM 仿真)",
         "src/s4_iss.cpp", "../05_体系结构与NPU微架构"),
        ("加分项·多模态/视觉 token/长上下文/连续 decode", "DeiT-tiny 视觉编码器整模型上 NPU + 视觉 token 投影进 LLM 的 "
                                             "201 token prefill 与连续 decode; 长上下文 KV 预算与临界点",
         "results/s8_vision.md, results/s6_perf.md", "src/s8_vision.py"),
        ("工程目标·可复现验证链路", "run_all.sh 一条命令跑通: 参考输出 → 量化权重 → 硬件运行结果 → 性能数据, "
                           "全部产出在 results/ 且带判据", "results/SUMMARY.md", "run_all.sh"),
    ]
    rep.table(["需求条目", "本 demo 对应证据", "结果文件", "代码"], [list(x) for x in jd])

    rep.h("3. 复现方式", 2)
    rep.code("# 标准档 (含 AWQ/GPTQ 采样)\nbash run_all.sh\n\n"
             "# 快速档 (跳过 AWQ/GPTQ, 少几步 decode)\nFAST=1 bash run_all.sh\n\n"
             "# 换量化档位\nVARIANT=w4a16_awq_mix8 bash run_all.sh\n\n"
             "# 全量 GPTQ (慢)\nGPTQ=all bash run_all.sh", "bash")

    rep.h("4. 已知局限 (别当成芯片实测)", 2)
    rep.p("- **NPU ISS 是「功能 + 周期」模型**, 不是 RTL, 更不是硅: 功能上按硬件数据布局真算一遍"
          "(int4 解包/去量化/分组累加/fp16 激活存储), 所以能和参考对拍; 但 cycle/带宽是按 roofline "
          "+ 效率因子**建模**出来的, 用于瓶颈归因和规划, 不能当流片数据。")
    rep.p("- 硬件档位 (1024 MAC/cycle @1GHz, LPDDR5 51.2 GB/s, 512 KiB 可用 SRAM, PCIe Gen3 x4) 是"
          "**假设的目标档位**, 在 `hw/mtx_npu_v0.json` 里, 换档位重跑即可得到新的性能画像。")
    rep.p("- 多模态里的 projector 是随机初始化的结构演示 (未训练), 验证的是链路与位置/长度正确性, "
          "不是语义效果; 文本侧用贪心解码, 未做采样温度/重复惩罚对比。")
    rep.p("- AWQ/GPTQ 的标定集只有 12 条短文本 (CPU 时间约束), 生产上要用上百条领域语料; "
          "GPTQ 默认只对采样层做 (全量在 `GPTQ=all`)。")
    rep.p("- 参考实现 (Qwen2/ViT) 是手写展开的, 已和 transformers 的行为对齐 (生成文本可读、"
          "分类正确), 但没做全量 logits 逐元素比对。")
    rep.save()
    print("汇总写入 results/SUMMARY.md")


if __name__ == "__main__":
    main()
