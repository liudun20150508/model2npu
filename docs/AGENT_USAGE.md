# 给智能体用：Skill 与 MCP

两种接入方式，能力等价：

| 方式 | 入口 | 适合 |
|---|---|---|
| Skill | [`SKILL.md`](../SKILL.md) | Codex / Claude 等支持 skill 的智能体，用自然语言触发 |
| MCP | [`mcp/server.py`](../mcp/server.py) | 任意 MCP 客户端（Claude Desktop、IDE 插件、自研 agent） |

## 1. Skill 模式

把仓库当技能包：智能体读到 `SKILL.md` 后，会按「阶段 → 产物 → 判据」的固定套路执行，
并在失败时按判据前缀定位到阶段。触发语示例：

- 「把 Qwen2.5-0.5B 适配到我们 NPU，给我对齐报告」
- 「w4a16 的生成结果和 fp32 不一致，帮我定位到层」
- 「帮我评估 AWQ 和 GPTQ 在我们 512KB SRAM 档位上的精度/存储对比」
- 「给这份硬件档位重跑性能画像，并判断是算力还是访存瓶颈」

## 2. MCP 模式

### 客户端配置（stdio）

```json
{
  "mcpServers": {
    "model2npu": {
      "command": "python3",
      "args": ["/abs/path/to/model2npu/mcp/server.py", "--stdio"]
    }
  }
}
```

### 工具清单

| 工具 | 作用 | 关键参数 |
|---|---|---|
| `env_check` | 检查 Python/g++/ISS/模型/CANN/npu-smi | — |
| `list_stages` | 列出阶段、说明与产物路径 | — |
| `run_stage` | 跑某个阶段 | `stage`（`s2_align`/`s3_quant`/`s5_e2e`…）、`args`、`timeout_s` |
| `list_rules` | 列出声明式判据规则（`rules/gates.json`） | `stage`（可选，只看某阶段） |
| `run_rules` | 跑规则引擎裁决（22 条判据 + 新鲜度），返回失败与过期清单 | `stage`、`json` |
| `verify_all` | 兼容入口：内部同样调用规则引擎 | — |
| `read_gate_table` | 读 `results/rules_report.json`（没有则读 `ALL_GATES.json`） | — |
| `read_report` | 读 `results/` 下报告 | `file`（默认 `SUMMARY.md`）、`max_chars` |
| `kb_list` | 列出知识库条目 | — |
| `kb_search` | 按关键词/症状检索知识库（定位经验） | `query` |
| `kb_get` | 读知识库条目全文 | `id`、`max_chars` |
| `ascend_build_om` | 真机：ONNX → ATC → `.om` | `onnx`、`out_prefix`、`input_shape` |
| `ascend_run_om` | 真机：pyACL 在 Ascend 上执行 `.om` | `om`、`device_id` |

MCP **resources**（`resources/list` / `resources/read`）：`kb://<id>` 读知识库全文，
`report://<文件名>` 读 `results/` 下的报告。

### 命令行自测（不需要 MCP 客户端）

```bash
python3 mcp/server.py --list-tools
python3 mcp/server.py --call env_check
python3 mcp/server.py --call list_stages
python3 mcp/server.py --call run_stage --args '{"stage":"s2_align"}'
python3 mcp/server.py --call list_rules --args '{"stage":"S5"}'
python3 mcp/server.py --call run_rules
python3 mcp/server.py --call kb_search --args '{"query":"top-1 飘 cosine 高"}'
python3 mcp/server.py --call kb_get --args '{"id":"quant-bit-exact"}'
python3 mcp/server.py --call read_report --args '{"file":"s7_regression.md","max_chars":3000}'
```

### 一个完整的智能体回合示例

```
user  : 帮我确认这套模型适配现在是不是全绿
agent : env_check                      -> 模型/编译器/ISS 都就绪
        run_rules                      -> 22/22 PASS, 无过期产物
        read_report(SUMMARY.md)        -> 汇报 TTFT/TPOT/对齐数字
agent : 结论: 全绿。若要换档位, 提供 hw/your_npu.json 后我重跑 run_all.sh。
```

失败回合：

```
agent : run_rules                      -> FAIL: e2e.logits_cosine_bc = 0.87
        kb_search("cosine 掉点 定位")   -> 命中 align-first-divergence
        run_stage(s7_regress)          -> 逐层 cosine 曲线, 第一次掉点在第 8 层
        run_stage(s2_align)            -> 该层算子用例明细
agent : 定位为 RoPE theta 配置错误 (注入实验同签名), 建议检查第 8 层 theta 参数
```

## 3. 智能体使用时的硬约束（已写进 SKILL.md）

1. `verify_all.py` 非 0 时不得对外声称「全绿」；
2. 不得把 ISA 仿真器的 cycle/带宽说成芯片实测；
3. 不得把未训练 projector 的多模态输出说成语义效果。

这三条对应的原始声明分别写在 `results/ALL_GATES.md`、`results/SUMMARY.md` 第 4 节、
`results/s8_vision.md` 第 4 节。
