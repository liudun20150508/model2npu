---
name: model2npu
description: >
  把 LLM / VLM（Qwen2.5-0.5B、DeiT-tiny 等）适配到 NPU 并做验证：算子级位级对齐、量化
  (RTN/AWQ/GPTQ) 与硬件权重打包、指令级仿真执行、端到端 logits 对齐、TTFT/TPOT 性能归因、
  回归门禁与故障定位、真机 Ascend (ATC/ACL) 闭环。当用户提出「模型上 NPU / 端侧部署」、
  「算子对齐 / 数值对不上 / 定位误差层」、「量化精度评估 / 选量化方案」、「性能瓶颈归因」、
  「FPGA/ASIC 验证判据」、「换一颗 NPU 档位重跑」时使用本技能。
---

# model2npu —— 模型适配与算子验证流水线

## 触发场景

1. 要把某个模型「搬上 NPU」，需要从参考实现对拍到设备结果；
2. 数值对不上，需要定位是**哪一层、哪个算子**开始偏离；
3. 要选量化方案（RTN / AWQ / GPTQ、分组大小、混合精度），需要统一口径的精度 + 存储对比；
4. 要性能数字（TTFT / TPOT / 访存量 / 带宽瓶颈）与瓶颈归因；
5. 要给芯片/FPGA 验证建立**可回归的判据**；
6. 要换一颗 NPU（换档位参数）重跑同一套判据。

## 使用前提

```bash
bash install.sh                 # 检查 Python(numpy/torch/tokenizers/pillow) + g++, 编译 ISS 与 C++ 算子
bash scripts/download_models.sh # 首次需下载模型 (Qwen2.5-0.5B-Instruct + deit-tiny-patch16-224)
```

若模型缺失、编译器缺失，脚本会明确报出缺什么，不要继续执行后续阶段。

## 标准流程（按顺序，每一步都可单独重跑）

| 步 | 命令 | 成功标志 | 产物 |
|---|---|---|---|
| 1 | `python3 src/s1_analyze_model.py` | 打印参数统计 | `results/s1_model_analysis.md` |
| 2 | `python3 src/s2_align.py` | `算子对齐: 全绿` | `results/s2_ops_alignment.md` |
| 3 | `python3 src/s3_quant.py --gptq sample` | 列出各量化档位 cosine | `results/s3_quant.md` |
| 4 | `python3 src/s3_pack.py --variant w4a16_rtn_mix8` | `bit 级自检` 通过 | `results/s3_pack.md` |
| 5 | `python3 src/s5_e2e.py --variant w4a16_rtn_mix8 --decode 8` | ISS cosine ≥ 0.999 | `results/s5_e2e.md` |
| 6 | `python3 src/s6_perf.py --variant w4a16_rtn_mix8` | 输出 TTFT/TPOT | `results/s6_perf.md` |
| 7 | `python3 src/s8_vision.py` | `视觉链路: OK` | `results/s8_vision.md` |
| 8 | `python3 src/s7_regress.py --allow-fail` | `11 PASS / 0 FAIL` | `results/s7_regression.md` |
| 9 | `python3 src/s9_report.py` | 生成汇总 | `results/SUMMARY.md` |
| 10 | `python3 src/verify_all.py` | `20/20 PASS`，退出码 0 | `results/ALL_GATES.md` |

整链一条命令：`bash run_all.sh`（快速档 `FAST=1 bash run_all.sh`）。

## 判据怎么读（重点）

判据是**声明式规则**（`rules/gates.json`），由 `src/rule_engine.py` 裁决 —— 不要再往脚本里塞 if：

```bash
python3 src/rule_engine.py            # 全部判据 + 产物新鲜度 -> results/RULES.md / rules_report.json
python3 src/rule_engine.py --only S5  # 只看某阶段
python3 src/verify_all.py             # 兼容入口, 产物名仍是 results/ALL_GATES.md
```

`results/rules_report.json` 结构：

```json
{"summary":{"n_pass":22,"n_error":0,"n_warn":0,"n_stale":0},
 "rules":[{"id":"e2e.logits_cosine_bc","value":0.999987,"limit":">= 0.999","ok":true,
           "severity":"error","evidence":"results/s5_e2e.md"}, ...],
 "freshness":[{"stage":"S5 端到端集成与数值对齐","ok":true,"stale_by":[]}]}
```

- `rules[].ok == false`：`severity=error` 必须报告并修复；`severity=warn` 需在报告里点出；
- `freshness[].stale_by != []` → 该阶段产物比源码旧，结果不是当前代码跑的，必须重跑该阶段；
- 退出码 = error 失败数 + 过期产物数，可直接作为 CI 门禁；
- 阈值要改就改 `rules/gates.json`，并说明理由（例如工程目标从 15 ms 放宽到 20 ms）。

## 失败时怎么定位（固定套路）

1. 先跑 `rule_engine.py`，拿到 FAIL 的判据 id；
2. 按判据前缀找阶段：`ops.*` → S2；`pack.*`/`quant.*` → S3；`e2e.*` → S5；`perf.*` → S6；
   `regress.*`/`repeat.*`/`debug.*` → S7；`vision.*` → S8；
3. 打开对应 `results/sX_*.json`，看明细数组里第一个不通过的用例/层；
4. **先查知识库**：`python3 mcp/server.py --call kb_search --args '{"query":"<症状关键词>"}'`
   （或 MCP 的 `kb_search` 工具）—— 常见症状（cosine 掉点、top-1 飘但 cosine 高、
   decode 崩、性能不达标、真机编译失败）都有对应条目与处置步骤；
5. 需要逐层定位时用 S7 的故障注入思路：同一链路只在一层注入一类错误，看逐层 cosine
   曲线第一次掉下去的位置 —— `results/s7_fault_injection.json` 里有现成的曲线结构可复用；
6. 视觉链路排查可参考 `python3 src/dbg_vit_align.py`（对比 ISS READBACK dump 与参考逐层结果）。

## 知识库（必须先用再下结论）

| id | 什么时候用 |
|---|---|
| `align-first-divergence` | 数值对不上、要定位到层 |
| `quant-bit-exact` | 量化后 top-1 飘、怀疑权重格式 |
| `perf-attribution` | 性能不达标、要先分清算力/访存 bound |
| `multimodal-kv` | 多模态或长上下文结果异常 |
| `realhw-bringup` | 真机图编译/执行失败 |

索引在 `knowledge/index.json`；MCP 侧还有资源 `kb://<id>` 可直接读全文。

## 换芯片 / 换后端

- 换档位（MAC/cycle、频率、SRAM、DRAM 带宽、PCIe、效率因子、算子固定开销）：
  复制 `hw/mtx_npu_v0.json` 改成自家参数 → `MTX_HW=hw/your_npu.json bash run_all.sh`，
  判据阈值不动，性能画像重新生成。
- 换后端：真机 Ascend 走 `src/ascend_build_om.sh` + `src/ascend_acl_run.py`
  （`ONNX → ATC → .om → pyACL 执行`）；换别家 ISA 时替换 `src/s4_iss.cpp` 的指令语义、
  保留 `s4_gen_program.py` 的微码模板与全部判据。

## 不许做的事

- 不要把自研 ISA 仿真的 cycle/带宽数字说成芯片实测 —— 报告的「已知局限」章节已声明；
- 不要在 `verify_all.py` 非 0 的情况下对外声称「全绿」；
- 不要把未训练的 projector 的多模态输出说成语义效果（`results/s8_vision.md` 已标注）。

## 等价的 MCP 工具

若通过 MCP 接入（`mcp/server.py`），工具与上面的命令一一对应：
`env_check`、`list_stages`、`run_stage`、`list_rules`、`run_rules`、`verify_all`、
`read_gate_table`、`read_report`、`kb_list`、`kb_search`、`kb_get`、
`ascend_build_om`、`ascend_run_om`；资源 `kb://<id>`、`report://<文件>`。
