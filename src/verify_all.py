#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""兼容入口: 判据裁决已迁移到规则引擎。

真正的裁决逻辑在 `rules/gates.json` (声明式规则) + `src/rule_engine.py` (引擎)。
本脚本保留旧名字与旧产物名, 方便已有脚本 / CI / 文档继续引用:

    results/RULES.md           -> 同时写一份 results/ALL_GATES.md
    results/rules_report.json  -> 同时写一份 results/ALL_GATES.json

退出码 = error 级失败数 + 过期产物数 (0 = 全绿)。
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
RESULTS = os.path.join(ROOT, "results")


def main():
    rc = subprocess.call([sys.executable, os.path.join(SRC, "rule_engine.py")] +
                         sys.argv[1:], cwd=ROOT)
    rp, ap_ = os.path.join(RESULTS, "RULES.md"), os.path.join(RESULTS, "ALL_GATES.md")
    if os.path.exists(rp):
        with open(rp, encoding="utf-8") as f:
            text = f.read()
        with open(ap_, "w", encoding="utf-8") as f:
            f.write(text.replace("# 判据裁决总表 (规则引擎)",
                                 "# 全链路判据总表 (规则引擎裁决, 兼容入口 verify_all.py)"))
    jp, jap = os.path.join(RESULTS, "rules_report.json"), os.path.join(RESULTS, "ALL_GATES.json")
    if os.path.exists(jp):
        shutil.copyfile(jp, jap)
    print("兼容产物: results/ALL_GATES.md / results/ALL_GATES.json")
    sys.exit(rc)


if __name__ == "__main__":
    main()
