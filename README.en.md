# model2npu

**A reproducible framework for adapting LLM/VLM models to an NPU/FPGA target and verifying them
end-to-end: one pipeline, a declarative rule engine, a searchable knowledge base, plus Skill and
MCP interfaces for agents.**

All measurements in this repo were produced on a **Huawei Ascend 910B3** server.
Machine configuration and timings are in [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md);
raw logs are kept in `results/logs/`.

## Why it exists

Getting a model onto a new NPU is mostly repetitive verification work: writing reference
implementations, aligning operators, hunting the layer where quantization error explodes,
checking that the packed weight format is right, and attributing latency to compute or memory.
model2npu turns that work into reusable infrastructure:

| Task | Naive approach | This repo |
|---|---|---|
| Evaluate a new chip/RTL revision | rewrite the benchmark scripts | copy `hw/mtx_npu_v0.json`, edit the target parameters (MAC/cycle, clock, SRAM, DRAM BW, PCIe, efficiency, per-op overhead), then `MTX_HW=hw/yours.json bash run_all.sh` — the acceptance criteria stay unchanged |
| Locate a numerical mismatch | manual layer-by-layer printing | single-fault injection + per-layer cosine curve; the first layer that diverges is the culprit |
| Choose a quantization scheme | one-off scripts per scheme | unified comparison of RTN / AWQ / GPTQ, group size, mixed precision, with per-tensor error CSV |
| Weight-format export | trust the final accuracy | custom int4 packing with a bit-exact unpack self-check |
| Review / handover | "it runs" | declarative gates in `rules/gates.json` + automated verdict (`src/rule_engine.py`), CI-ready |

## Four building blocks

1. **Rule engine** — all 22 acceptance criteria live in `rules/gates.json` as declarative rules
   (`expr`, `op`, `threshold`, `severity`, `evidence`), evaluated by `src/rule_engine.py` with
   restricted expression evaluation, three severity levels, and artifact-freshness checks
   (so a stale result can never be reported as green). Exit code = error-severity failures +
   stale artifacts.
2. **Knowledge base** — `knowledge/index.json` plus `knowledge/kb_*.md`: five searchable entries
   (first-divergence localization, bit-exact packing, performance attribution, multimodal/KV
   pitfalls, real-hardware bring-up).
3. **Skill** — [`SKILL.md`](SKILL.md) tells an agent when to trigger, which stage to run, how to
   read the verdict JSON, how to localize failures, and what it must not claim.
4. **MCP** — `mcp/server.py` (stdio, no third-party deps) exposes tools (`env_check`,
   `list_stages`, `run_stage`, `list_rules`, `run_rules`, `verify_all`, `read_gate_table`,
   `read_report`, `kb_list`, `kb_search`, `kb_get`, `ascend_build_om`, `ascend_run_om`) and
   resources (`kb://<id>`, `report://<file>`).

## Pipeline (10 stages)

S1 model analysis & deployment planning (per-op roofline, SRAM tiling, KV budget) →
S2 operator alignment (Python ↔ C++ ↔ PyTorch) → S3 quantization (RTN/AWQ/GPTQ/mixed) →
S3b int4 packing & device image (bit-exact self-check) → S4a microcode generation →
S4b instruction-level simulation (real math + modeled cycles/DRAM/PCIe/SRAM) →
S5 end-to-end three-way alignment (prefill + continuous decode) → S6 performance attribution →
S7 regression gates & fault localization → S8 vision encoder plus vision tokens into the LLM →
S9 summary. `bash run_all.sh` runs all of them (~12.5 min on the machine above).

## Measured results (22/22 gates PASS)

- ISS vs quantized reference: logits cosine 0.999987, generated tokens fully consistent;
- TTFT 9.36 ms, TPOT 7.89 ms (126.7 tok/s single stream), 323 MiB DRAM per decode token;
- six quantization presets; GPTQ cuts per-layer output error to 0.685x of RTN;
- device image 312.72 MiB (2.92x smaller than fp16), metadata 4.01%, unpack verified bit-exact;
- 12 structural + 13 quantized operator cases; 4120 fp16-rounding cases bit-exact;
- fault injection 3/3 localized to the injected layer; two runs bit-identical;
- DeiT-tiny top-1 matches fp32 (cosine 0.9823); 201-token multimodal sequence runs through
  device prefill + continuous decode;
- real hardware: `ONNX → ATC → .om` (30 s) then `pyACL` execution on Ascend 910B3 (4 s).

## Quick start

```bash
bash install.sh                  # env check + build ISS/C++ ops + smoke test (no model needed)
bash scripts/download_models.sh  # Qwen2.5-0.5B-Instruct + deit-tiny-patch16-224
bash run_all.sh                  # full pipeline, ~12.5 min (FAST=1 for a quick pass)
python3 src/rule_engine.py       # acceptance verdict -> results/RULES.md
```

## Honest limitations

The device side is a self-defined instruction-level simulator (functional math on the real
hardware data layout plus modeled cycles/bandwidth), not silicon. `hw/mtx_npu_v0.json` is an
assumed target specification. On real hardware only a mobile-net-scale minimal loop is verified.
The multimodal projector is randomly initialized — it validates the data path, not semantics.
Section 9 of the Chinese README carries the full statement.

MIT licensed (see [`LICENSE`](LICENSE)); model weights are downloaded separately under their own
licenses.
