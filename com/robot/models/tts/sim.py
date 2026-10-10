# -*- coding: utf-8 -*-
"""PC 仿真: fp 与 int8 量化 RKNN 模型均可 init_runtime()(target=None) 仿真运行"""
import argparse, json
import numpy as np
from rknn.api import RKNN

OUT_NAMES = {
    'enc': ['logw', 'x_mask', 'g', 'm_p', 'logs_p'],
    'dec': ['y'],
}

def load_inputs(npz_path, manifest):
    data = np.load(npz_path)
    ins = []
    for meta in manifest:
        a = np.asarray(data[meta['name']])
        a = a.astype(np.int32 if meta.get('dtype') == 'INT32' else np.float32)
        ins.append(np.ascontiguousarray(a))
    return ins

def run_sim(rknn_path, inputs, do_analysis=False):
    rknn = RKNN(verbose=False)
    assert rknn.load_rknn(rknn_path) == 0
    assert rknn.init_runtime() == 0          # target=None -> PC 仿真
    outputs = rknn.inference(inputs=inputs)
    if do_analysis:
        rknn.accuracy_analysis(inputs=inputs, target=None)  # 逐层量化误差
    rknn.release()
    return [np.asarray(o, np.float32) for o in outputs]

def compare(a, b, name):
    a, b = a.reshape(-1), b.reshape(-1)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    err = float(np.max(np.abs(a - b)))
    print(f'  {name:8s}  cos={cos:.6f}  max_abs_err={err:.4e}  (range {a.min():.3f}~{a.max():.3f})')
    return cos

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fp', help='fp rknn 模型')
    ap.add_argument('--quant', help='int8 rknn 模型')
    ap.add_argument('--golden', required=True, help='gen_calib.py 生成的 golden npz')
    ap.add_argument('--manifest', required=True, help='gen_calib.py 生成的 manifest json')
    ap.add_argument('--kind', choices=['enc', 'dec'], required=True)
    ap.add_argument('--analysis', action='store_true', help='量化模型逐层精度分析')
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    inputs = load_inputs(args.golden, manifest)
    names = OUT_NAMES[args.kind]

    outs = {}
    if args.fp:
        print(f'==> 仿真 fp 模型: {args.fp}')
        outs['fp'] = run_sim(args.fp, inputs)
    if args.quant:
        print(f'==> 仿真 int8 模型: {args.quant}')
        outs['quant'] = run_sim(args.quant, inputs, do_analysis=args.analysis)

    if 'fp' in outs and 'quant' in outs:
        print('==> fp vs int8 (量化精度损失):')
        for i, n in enumerate(names):
            compare(outs['fp'][i], outs['quant'][i], n)
        print('参考: cos > 0.999 基本无损; < 0.99 建议混合量化/int16')

    if 'fp' in outs:
        for i, n in enumerate(names):
            np.save(f'sim_fp_out_{n}.npy', outs['fp'][i])
        print('fp 仿真输出已保存 sim_fp_out_*.npy (可与 onnxruntime 输出人工对比)')# -*- coding: utf-8 -*-
