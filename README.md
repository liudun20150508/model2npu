# model2npu

**把大模型（LLM / VLM）适配到 NPU 并验证到底的工具框架：一条可复现的流水线 + 一套声明式判据
+ 一个可被智能体调用的知识库。**

> 面向端侧大模型推理 NPU 与 FPGA 原型。目标模型：Qwen2.5-0.5B-Instruct + DeiT-tiny。
> 全部实测在 **华为昇腾（Ascend）910B3** 服务器上完成，机器配置与耗时见
> [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)，原始日志在 `results/logs/`。

---

## 1. 这套框架能帮迈特芯加速什么

把模型搬上自家 NPU/FPGA 的时间，大头不在算法，而在**重复的验证劳动**：
写参考实现、逐算子对拍、查量化误差从哪一层开始变大、确认权重打包格式没写错、
把性能数字归因到算力还是访存。每个模型、每一版 RTL 都要再做一遍。

model2npu 把这段工作固化成**可参数化、可回归、可被智能体驱动**的流水线，具体加速点：

| 迈特芯的常见环节 | 原做法 | 用本框架 |
|---|---|---|
| 换一颗芯片/一版 RTL 的性能评估 | 重写评测脚本 | 复制 `hw/mtx_npu_v0.json` 改档位参数（MAC/周期、频率、SRAM、带宽、PCIe、效率因子、算子固定开销），`MTX_HW=hw/你们的.json bash run_all.sh` 得到新画像，**判据不变** |
| 算子数值对不上 | 人工逐层打印对比 | 同链路单点注入 + 逐层 cosine 曲线，直接给出「第一次掉点在哪个算子」 |
| 量化方案选型 | 各写一套评测 | RTN / AWQ / GPTQ / 分组 / 混合精度统一口径出精度 + 存储 + 逐 tensor 误差 CSV |
| 权重格式导出 | 只能看最终精度 | 自定义打包格式 + 解包**逐 bit 自检**，把格式错误挡在算子级 |
| 交付/评审 | 口头说「跑通了」 | `rules/gates.json` 声明式判据 + `rule_engine.py` 自动裁决，一条命令出结论（可挂 CI） |
| 新人接手 / 跨团队协作 | 靠人带 | `knowledge/` 知识库沉淀踩坑经验，`SKILL.md` + MCP 让智能体按同一套方法论干活 |

一句话：**把「模型适配 + 算子验证 + 性能归因」从一次性劳动变成可复用的基础设施。**

## 2. 四件套（框架结构）

```
                    ┌───────────────────────── 智能体接入 ─────────────────────────┐
                    │  SKILL.md (技能)            mcp/server.py (MCP 工具 + 资源)   │
                    └───────────────┬─────────────────────────┬───────────────────┘
                                    ▼                         ▼
   ┌── 规则引擎 ──────────┐   ┌── 流水线 (10 阶段) ──┐   ┌── 知识库 ──────────┐
   │ rules/gates.json     │   │ S1..S9 + 真机链路     │   │ knowledge/*.md      │
   │ src/rule_engine.py   │◀──│ run_all.sh           │──▶│ knowledge/index.json│
   │ 22 条判据 + 新鲜度    │   │ results/*.json       │   │ 可检索经验条目       │
   └──────────────────────┘   └──────────────────────┘   └────────────────────┘
```

### 2.1 规则引擎（判据与代码分离）

判据不再是脚本里的 `if`，而是声明式规则 `rules/gates.json`：

```json
{
  "id": "e2e.logits_cosine_bc", "stage": "S5", "severity": "error",
  "desc": "ISS vs 量化模拟 (prefill logits cosine)",
  "expr": "s5.metrics_bc.logits_cosine_prefill", "op": ">=", "threshold": 0.999,
  "evidence": "results/s5_e2e.md"
}
```

```bash
python3 src/rule_engine.py            # 22 条判据 + 11 条新鲜度规则 -> results/RULES.md
python3 src/rule_engine.py --only S5  # 只看某阶段
python3 src/rule_engine.py --json     # 机器可读摘要 (可接 CI)
```

能力：受限表达式求值（不允许 import/写文件）、三级严重度（error 影响退出码 / warn 提示 /
info 记录）、**产物新鲜度检查**（防止拿旧结果当结论）、退出码 = error 失败数 + 过期产物数。

### 2.2 知识库（踩坑经验可检索）

`knowledge/index.json` 是索引，每条含 `id / tags / stage / symptoms / verdict`，
全文在 `knowledge/kb_*.md`。当前条目：

| id | 一句话结论 |
|---|---|
| `align-first-divergence` | 数值对不上时，找「逐层 cosine **第一次**掉点」的那一层，而不是误差最大层 |
| `quant-bit-exact` | cosine 0.9+ 不代表权重格式对，必须做解包逐 bit 自检 |
| `perf-attribution` | 先判算力 bound 还是访存 bound；效率因子未校准前，性能数字只能当量级参考 |
| `multimodal-kv` | 多模态/长上下文问题多半在 KV 布局与位置 id，不在数值精度 |
| `realhw-bringup` | 真机 bring-up 拆四段验证；环境坑（编译器的 Python 依赖、框架与硬件插件版本）先于算法坑 |

### 2.3 Skill（智能体技能入口）

[`SKILL.md`](SKILL.md)：定义触发场景、阶段命令表、判据读法、失败定位套路、硬约束
（不得把仿真数字说成芯片实测等）。支持 skill 的智能体可直接按它执行。

### 2.4 MCP（工具 + 资源）

`mcp/server.py`，stdio、零第三方依赖：

```bash
python3 mcp/server.py --list-tools
python3 mcp/server.py --call env_check
python3 mcp/server.py --call run_rules --args '{"stage":"S5"}'
python3 mcp/server.py --call kb_search --args '{"query":"top-1 飘 但 cosine 高"}'
```

| 类别 | 工具 |
|---|---|
| 环境/阶段 | `env_check`、`list_stages`、`run_stage` |
| 判据 | `list_rules`、`run_rules`、`verify_all`、`read_gate_table`、`read_report` |
| 知识库 | `kb_list`、`kb_search`、`kb_get` |
| 真机 | `ascend_build_om`、`ascend_run_om` |

另暴露 MCP **resources**：`kb://<id>`（知识库全文）、`report://<文件>`（阶段报告）。

## 3. 本机实测环境（华为昇腾）

| 项 | 配置 |
|---|---|
| NPU | **Ascend 910B3** × 1 chip（NPU id 1，Bus-Id `0000:C2:00.0`），HBM 64 GB，`npu-smi` Health OK，实测 ~92 W / 53 °C |
| NPU 软件栈 | 驱动 26.1.1（ascendhal 7.35.23）+ **CANN 9.1.0**（`atc` / `pyACL` 可用） |
| CPU | 鲲鹏 Kunpeng-920（aarch64），256 逻辑核，1 thread/core，主频上限 3.0 GHz |
| 内存 | 502 GB（量化阶段单进程 RSS 峰值 ~10.5 GB / 127 线程） |
| OS | Ubuntu 24.04.4 LTS，kernel 5.10.0-216.0.0.115.oe2203sp4.aarch64 |
| 工具链 | g++ 13.3.0；Python 3.11.15；torch 2.11.0 / torch_npu 2.10.0.post4（**两者 ABI 不匹配**，真机链路走 CANN 原生 acl） |

### 实测结果（`results/RULES.md`：22/22 PASS）

| 维度 | 实测 |
|---|---|
| 全链路 | `run_all.sh` 12 步全 OK，**约 12.5 分钟**（最慢 S3 量化 601 s） |
| 端到端性能 | TTFT **9.36 ms**，TPOT **7.89 ms**（126.7 tok/s，单路），decode 访存 323 MiB/token |
| 三端对齐 | ISS vs 量化模拟 logits cosine **0.999987**，连续 decode 生成 token 全一致 |
| 量化 | 6 档：fp16 0.999993 / W8A16 0.999432 / RTN-g128 0.942823 / RTN-mix8 0.943227 / AWQ 0.980592 / GPTQ 0.964906（vs fp32 logits cosine） |
| 量化收益 | GPTQ 层输出误差降到同 group RTN 的 **0.685 倍**（全部矩阵更优）；AWQ 中位 0.911、80.4% tensor 更优 |
| 权重打包 | 自定义 int4 设备镜像 **312.72 MiB**（fp16 942 MiB 的 2.92×），元数据 4.01%，解包**逐 bit 一致** |
| 算子对齐 | 12 结构算子 + 13 量化算子用例；fp16 舍入 4120 例位级一致 100% |
| 回归门禁 | 11 PASS / 0 FAIL；故障注入 3/3 定位到注入层；同微码两次运行逐 bit 一致 |
| 视觉 + 多模态 | DeiT-tiny top-1 与 fp32 一致（cosine 0.9823）；197 视觉 token + 4 文本 = 201 序列走通设备 prefill + 连续 decode，KV 峰值 106.5 KiB |
| 真机闭环 | Ascend 910B3：`ONNX → ATC → .om`（30 s，9.5 MB）→ `pyACL` 设备执行（4 s，1000 类输出） |

## 4. 十个阶段

| 阶段 | 做什么 | 产物 |
|---|---|---|
| S1 模型分析与部署规划 | 逐算子 roofline（算力 vs 访存）、算子映射、SRAM 分块、KV/权重/激活占用、长上下文临界点 | `results/s1_model_analysis.md` |
| S2 算子适配与验证 | Python 参考 ↔ C++ native ↔ PyTorch 真值三方对齐；形状/精度/边界 | `results/s2_ops_alignment.md` |
| S3 量化与精度优化 | RTN / AWQ / GPTQ、对称与非对称、g32/g128、混合精度、逐 tensor 误差 | `results/s3_quant.md` |
| S3b 权重打包与镜像 | 自定义 int4 打包 + 设备镜像 + 解包逐 bit 自检 | `results/s3_pack.md` |
| S4a 微码生成 | 模型算子 → ISA 指令，分块参数取自 S1 规划 | `build/npu_program_*.mtx` |
| S4b ISS 执行 | 指令级仿真：真算数值 + 建模 cycle/DRAM/PCIe/SRAM 峰值 | `results/npu_profile_*.json` |
| S5 端到端对齐 | 参考 / 量化模拟 / 设备三端 logits 与生成结果对拍（prefill + 连续 decode） | `results/s5_e2e.md` |
| S6 性能归因 | TTFT/TPOT/吞吐/带宽/存储，逐算子 cycle 与访存归因 | `results/s6_perf.md` |
| S7 回归门禁与故障定位 | 门禁判据 + 故障注入 → 逐层误差曲线 → 定位到层 | `results/s7_regression.md` |
| S8 视觉 + 多模态 | DeiT-tiny 整模型上设备 + 视觉 token 进 LLM 的 201 序列链路 | `results/s8_vision.md` |
| S9 汇总 | 一页纸结论 + 需求覆盖矩阵 | `results/SUMMARY.md` |

## 5. 安装与快速开始

```bash
bash install.sh                  # 环境自检 + 编译 ISS/C++ 算子 + 冒烟(S2, 无需模型)
bash scripts/download_models.sh  # Qwen2.5-0.5B-Instruct + deit-tiny-patch16-224 (~1.1 GB)
bash run_all.sh                  # 全链路 (约 12.5 分钟); 快速档 FAST=1 bash run_all.sh
python3 src/rule_engine.py       # 判据裁决 -> results/RULES.md (任一 error/过期 => 非 0)
bash deploy.sh                   # 上面几步串起来的一键部署
```

单阶段、换档位、真机：

```bash
python3 src/s2_align.py                                  # 算子对齐
python3 src/s3_quant.py --gptq sample                    # 量化 (sample|all|none)
python3 src/s5_e2e.py --variant w4a16_rtn_mix8 --decode 8
MTX_HW=hw/your_npu.json bash run_all.sh                  # 换硬件档位
bash src/ascend_build_om.sh model.onnx /tmp/out "input:1,3,224,224"   # 真机: 编译
python3 src/ascend_acl_run.py /tmp/out.om 0                           # 真机: 执行
```

## 6. 迁移到你们的 NPU / FPGA

详见 [`docs/PORTING.md`](docs/PORTING.md)。三层解耦，只有两层需要按芯片替换：

| 层 | 换什么 | 判据层 |
|---|---|---|
| 性能模型层 | `hw/*.json` 档位参数（**效率因子必须用 micro-benchmark 或 RTL 仿真校准**） | 不变 |
| 实现层 | `src/s4_iss.cpp` 的指令语义 / 权重格式 / 数据布局；或直接换成你们的 RTL + UVM / 自家 ISS | 不变 |
| 判据层 | `rules/gates.json` 只调阈值（若你们的工程目标不同） | —— |

验收清单（可直接当交付验收表）见 `docs/PORTING.md` 末节。

## 7. 目录结构

```
model2npu/
├── README.md / README.en.md     本文件 / 英文版
├── SKILL.md                     智能体技能入口 (触发场景/命令表/判据读法/硬约束)
├── rules/gates.json             规则引擎的声明式判据 (22 条 + 新鲜度规则)
├── knowledge/                   知识库: index.json + kb_*.md (踩坑经验全文)
├── mcp/server.py                MCP server (stdio; 工具 + 资源; 零第三方依赖)
├── install.sh / deploy.sh       环境安装 / 一键部署
├── scripts/download_models.sh   模型下载 (HF 镜像)
├── docs/                        ENVIRONMENT(环境实测) / PORTING(迁移) / AGENT_USAGE(智能体)
├── hw/mtx_npu_v0.json           硬件档位 (换 json 即换芯片假设)
├── src/
│   ├── rule_engine.py           规则引擎 (受限表达式求值 + 新鲜度检查)
│   ├── verify_all.py            兼容入口 (旧名 -> 规则引擎, 产物名不变)
│   ├── mtx_common.py / mtx_ops_ref.py / mtx_qwen.py / mtx_vit.py
│   ├── s1..s9 阶段脚本 · s2_ops_native.cpp · s4_iss.cpp (指令级仿真器)
│   ├── dbg_vit_align.py         视觉链路逐层诊断
│   └── ascend_build_om.sh · ascend_acl_run.py   真机 ATC/ACL 链路
├── run_all.sh                   全链路一键复跑
└── results/                     各阶段报告 (json 为机器可读原始数据) + 运行日志
```

## 8. 判据总表（22 条，全 PASS）

`python3 src/rule_engine.py` 实时生成；摘录关键条目：

| 判据 | 实测 | 门槛 | 严重度 |
|---|---|---|---|
| `ops.struct` / `ops.struct.bit_exact` | 12/12 通过；位级 100% | 全部通过 | error |
| `ops.quant` | 13/13，max rel_err 7.6e-06 | ≤ 1e-5 | error |
| `quant.variants` / `quant.gptq_gain` / `quant.awq_gain` | 6 档；0.685×；0.911× | 覆盖 + 优于 RTN | error / warn |
| `pack.bit_exact` / `pack.image` | 100%；312.72 MiB，元数据 4.01% | 100% / < 10% | error |
| `e2e.logits_cosine_bc` / `e2e.gen_match_bc` | 0.999987；全一致 | ≥ 0.999 / 全一致 | error |
| `e2e.logits_cosine_ab` / `e2e.top1_ab` | 0.9432 / 0.7143 | ≥ 0.9 / ≥ 0.6 | error / warn |
| `perf.ttft_ms` / `perf.tpot_ms` | 9.36 / 7.89 ms | ≤ 50 / ≤ 15 ms | error |
| `regress.all_pass` / `repeat.deterministic` / `debug.localize` | 0 FAIL；1.0；3/3 | 见规则文件 | error |
| `vision.align` / `vision.top1` / `vision.multimodal` | 0.9823；一致；全有限 | ≥ 0.95 / 一致 / 无不健全 | error |
| `realhw.acl_exec` | 有真机执行日志 | 存在 | warn |

改阈值请编辑 `rules/gates.json`，不要改脚本 —— 这是本框架的核心约定。

## 9. 诚实声明与局限

1. **自研 ISA 仿真器不是硅**：功能上按硬件数据布局真算一遍（int4 解包/去量化/分组累加/
   fp16 激活存储），因此能与参考逐位对拍；cycle/带宽是按 roofline + 效率因子**建模**的，
   用于瓶颈归因与规划，**不能当流片数据**。
2. `hw/mtx_npu_v0.json` 是**假设的目标档位**（1024 MAC/cycle @1 GHz、LPDDR5 51.2 GB/s、
   512 KiB 可用 SRAM、PCIe Gen3 x4），不是任何真实芯片的实测参数。
3. 真机只完成 **mobilenetv2 最小闭环**；把 Qwen2.5-0.5B / DeiT 搬上真机还需
   ONNX 导出 → ATC 图编译（采样算子需自写）→ 接 KV Cache 与 bf16/int8 权重格式。
4. 多模态的 projector 为**随机初始化**结构演示，验证链路与布局，不代表语义效果。
5. AWQ/GPTQ 标定集为 12 条中英混合短文本；换标定集所有数字会变，故判据用**相对关系**
   （优于同 group RTN）而非绝对阈值。

## 10. Roadmap

- [ ] Qwen2.5-0.5B 真机闭环（ONNX 导出 + ATC 图编译 + 自定义采样算子）
- [ ] ISS 支持多核/多 tile 与 NoC 建模（对齐片上互联档位）
- [ ] 更多模型档位：Qwen2.5-1.5B/7B、CLIP/SigLIP 视觉塔
- [ ] GitHub Actions：S2 算子对齐可在无模型环境下跑通（含规则引擎裁决）
- [ ] ONNX / TVM 后端适配器，复用同一套判据与知识库

## 11. 许可

MIT License，见 [`LICENSE`](LICENSE)。模型权重遵循各自原始许可，本仓库只提供下载脚本，
不随包分发。
