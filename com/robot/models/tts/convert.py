from rknn.api import RKNN
if __name__ == '__main__':
    platform = 'rk3566'
    # model_path, platform, do_quant, output_path = 'encoder-ZH_MIX_EN.onnx',platform,False,f'encode{platform}.rknn'
    model_path, platform, do_quant, output_path = 'decoder-ZH_MIX_EN.onnx',platform,False,f'decode{platform}.rknn'

    # Create RKNN object
    rknn = RKNN(verbose=False)
    # Pre-process config
    print('--> Config model')
    rknn.config(target_platform=platform)
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=do_quant)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Release
    rknn.release()