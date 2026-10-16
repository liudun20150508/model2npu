#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""迈特芯 · 模型适配与算子验证 Demo 公共工具.

目录约定:
  ROOT/           demo 根目录
  ROOT/src        源码
  ROOT/hw         硬件规格 (MTX-NPU v0 假设档位)
  ROOT/models     下载的真实权重 (Qwen2.5-0.5B-Instruct / deit-tiny)
  ROOT/build      C++ 编译产物
  ROOT/results    所有产出 (json / md / csv / 日志)
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
HW = os.path.join(ROOT, "hw")
MODELS = os.path.join(ROOT, "models")
BUILD = os.path.join(ROOT, "build")
RESULTS = os.path.join(ROOT, "results")
DOCS = os.path.join(ROOT, "docs")
LOGS = os.path.join(RESULTS, "logs")

QWEN_DIR = os.path.join(MODELS, "qwen2.5-0.5b-instruct")
VIT_DIR = os.path.join(MODELS, "deit-tiny-patch16-224")


def ensure_dirs():
    for d in (BUILD, RESULTS, LOGS):
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------- JSON / IO
def jdump(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=False)
    return path


def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def wtext(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text if text.endswith("\n") else text + "\n")
    return path


def hw_cfg():
    """MTX-NPU v0 硬件规格 (可用环境变量 MTX_HW 覆盖到别的档位)."""
    p = os.environ.get("MTX_HW", os.path.join(HW, "mtx_npu_v0.json"))
    return jload(p)


def sha1_file(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def size_of(path):
    return os.path.getsize(path) if os.path.exists(path) else 0


def human_bytes(n):
    n = float(n)
    for u in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or u == "GiB":
            return f"{int(n)} B" if u == "B" else f"{n:.2f} {u}"
        n /= 1024.0


def run(cmd, env=None, cwd=None, check=True, timeout=None):
    """跑子进程, 返回 (returncode, stdout+stderr)."""
    e = dict(os.environ)
    e.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    if env:
        e.update(env)
    p = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True,
                       text=True, env=e, cwd=cwd or ROOT, timeout=timeout)
    out = (p.stdout or "") + (p.stderr or "")
    if check and p.returncode != 0:
        raise RuntimeError(f"命令失败({p.returncode}): {cmd}\n{out[-4000:]}")
    return p.returncode, out


def cxx():
    return os.environ.get("CXX", "g++")


def compile_cpp(src, out, extra=(), std="c++17", omp=True, quiet=False):
    """编译单文件 C++ (O3 + OpenMP)."""
    cmd = [cxx(), f"-std={std}", "-O3", "-march=native"]
    if omp:
        cmd.append("-fopenmp")
    cmd += ["-o", out, src, *extra]
    rc, log = run(cmd, check=False)
    if rc != 0:
        raise RuntimeError(f"编译失败: {src}\n{log[-4000:]}")
    if not quiet:
        print(f"  [build] {os.path.basename(src)} -> {os.path.relpath(out, ROOT)}")
    return out


# ---------------------------------------------------------------- 报告工具
class Report:
    """一边攒 md 表格, 一边存 json."""

    def __init__(self, title, json_path=None, md_path=None):
        self.title = title
        self.lines = [f"# {title}", ""]
        self.json_path = json_path
        self.md_path = md_path
        self.data = {}
        self.t0 = time.time()

    def h(self, text, level=2):
        self.lines += ["#" * level + " " + text, ""]
        return self

    def p(self, text=""):
        self.lines += [text, ""]
        return self

    def kv(self, key, value):
        self.data[key] = value
        return self

    def table(self, headers, rows):
        rows = [[("" if c is None else str(c)) for c in r] for r in rows]
        w = [len(h) for h in headers]
        for r in rows:
            for i, c in enumerate(r):
                w[i] = max(w[i], len(c))
        self.lines.append("| " + " | ".join(h.ljust(w[i]) for i, h in enumerate(headers)) + " |")
        self.lines.append("|" + "|".join("-" * (x + 2) for x in w) + "|")
        for r in rows:
            self.lines.append("| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(r)) + " |")
        self.lines.append("")
        return self

    def code(self, text, lang=""):
        self.lines += [f"```{lang}", text.rstrip("\n"), "```", ""]
        return self

    def save(self, extra_note=None):
        self.data.setdefault("title", self.title)
        self.data["elapsed_s"] = round(time.time() - self.t0, 2)
        if extra_note:
            self.data["note"] = extra_note
        if self.json_path:
            jdump(self.data, self.json_path)
        if self.md_path:
            wtext(self.md_path, "\n".join(self.lines))
        return self.data


def banner(step, text):
    print(f"\n{'=' * 78}\n[{step}] {text}\n{'=' * 78}", flush=True)


def step_path(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


if __name__ == "__main__":
    ensure_dirs()
    print("ROOT   :", ROOT)
    print("python :", sys.version.split()[0], sys.executable)
    print("hw spec:", json.dumps(hw_cfg(), ensure_ascii=False)[:160])
