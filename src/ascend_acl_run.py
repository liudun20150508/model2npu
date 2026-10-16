#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ascend 真机链路 (2/2): 在 910B3 上执行 .om (pyACL 直连, 不依赖 torch/torch_npu)。

用法: python src/ascend_acl_run.py <model.om> [device_id]
说明: 本机 torch 2.11.0 与 torch_npu 2.10.0.post4 的 C++ ABI 不匹配, torch_npu 导入即失败,
      所以真机验证走 CANN 原生 acl, 绕开 PyTorch。
"""
import sys

import numpy as np

import acl

OM = sys.argv[1] if len(sys.argv) > 1 else "mbv2_910b.om"
DEV = int(sys.argv[2]) if len(sys.argv) > 2 else 0
H2D, D2H = 1, 2          # acl.rt.memcpy 的 kind


def ck(ret, what):
    if ret != 0:
        raise RuntimeError("%s -> %s (%s)" % (what, ret, acl.get_recent_err_msg()))


ck(acl.init(), "acl.init")
ck(acl.rt.set_device(DEV), "set_device")
print("device %d soc: %s" % (DEV, acl.get_soc_name()))

model_id, ret = acl.mdl.load_from_file(OM)
ck(ret, "load_from_file")
desc = acl.mdl.create_desc()
ck(acl.mdl.get_desc(desc, model_id), "get_desc")
n_in, n_out = acl.mdl.get_num_inputs(desc), acl.mdl.get_num_outputs(desc)
print("inputs %d outputs %d" % (n_in, n_out))


def make_ds(n, size_of):
    ds = acl.mdl.create_dataset()
    bufs = []
    for i in range(n):
        size = size_of(i)
        ptr, ret = acl.rt.malloc(size, 1)              # 1 = HUGE_FIRST
        ck(ret, "malloc")
        buf = acl.create_data_buffer(ptr, size)
        _, ret = acl.mdl.add_dataset_buffer(ds, buf)
        ck(ret, "add_dataset_buffer")
        bufs.append((ptr, size))
    return ds, bufs


in_ds, in_bufs = make_ds(n_in, lambda i: acl.mdl.get_input_size_by_index(desc, i))
out_ds, out_bufs = make_ds(n_out, lambda i: acl.mdl.get_output_size_by_index(desc, i))

host_in = np.linspace(-1.0, 1.0, in_bufs[0][1] // 4, dtype=np.float32)   # 演示输入
ck(acl.rt.memcpy(in_bufs[0][0], in_bufs[0][1], host_in.ctypes.data, host_in.nbytes, H2D), "H2D")
ck(acl.mdl.execute(model_id, in_ds, out_ds), "execute")

for i, (ptr, size) in enumerate(out_bufs):
    host_out = np.zeros(size // 4, dtype=np.float32)
    ck(acl.rt.memcpy(host_out.ctypes.data, size, ptr, size, D2H), "D2H")
    top5 = np.argsort(-host_out)[:5]
    print("output[%d] %d floats, top5 idx %s logits %s" %
          (i, host_out.size, top5.tolist(), np.round(host_out[top5], 4).tolist()))

acl.mdl.unload(model_id)
acl.rt.reset_device(DEV)
acl.finalize()
print("ACL 真机执行 OK")
