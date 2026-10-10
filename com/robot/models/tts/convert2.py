# -*- coding: utf-8 -*-
"""ONNX -> RKNN: fp 模型(PC仿真验证) + int8 量化模型(板端 NPU)"""
import argparse
from rknn.api import RKNN

def convert(model_path, platform, do_quant, dataset, output_path):
    rknn = RKNN(verbose=False)
    print('--> Config model')
    if rknn.config(target_platform=platform) != 0:
        exit(-1)
    print(f'--> Loading model: {model_path}')
    if rknn.load_onnx(model=model_path) != 0:
        print('Load model failed!'); exit(-1)
    print(f'--> Building model (do_quantization={do_quant})')
    if do_quant:
        ret = rknn.build(do_quantization=True, dataset=dataset)
    else:
        ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!'); exit(-1)
    print(f'--> Export rknn model: {output_path}')
    if rknn.export_rknn(output_path) != 0:
        print('Export failed!'); exit(-1)
    rknn.release()
    print('done')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='onnx 路径')
    ap.add_argument('--mode', choices=['fp', 'quant', 'both'], default='both')
    ap.add_argument('--dataset', default=None, help='量化校准集 txt (gen_calib.py 生成)')
    ap.add_argument('--platform', default='rk3566')
    ap.add_argument('--tag', default=None, help='输出文件前缀, 默认取 onnx 文件名')
    args = ap.parse_args()

    tag = args.tag or args.model.replace('.onnx', '')
    if args.mode in ('fp', 'both'):
        convert(args.model, args.platform, False, None, f'{tag}_fp.rknn')
    if args.mode in ('quant', 'both'):
        assert args.dataset, '量化模式必须提供 --dataset'
        convert(args.model, args.platform, True, args.dataset, f'{tag}_i8.rknn')