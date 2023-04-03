import torchvision
from ppq import *
from ppq.api import *
from Utilities.Imagenet import (evaluate_mmlab_module_with_imagenet,
                                evaluate_onnx_module_with_imagenet,
                                evaluate_ppq_module_with_imagenet,
                                evaluate_torch_module_with_imagenet,
                                load_imagenet_from_directory)
from ppq import TensorQuantizationConfig as TQC

"""
    使用这个脚本来测试量化 torchvision 中的典型分类模型
        使用 imagenet 中的数据测试量化精度与 calibration
        默认的 imagenet 数据集位置: Assets/Imagenet_Train, Assets/Imagenet_Valid
        你可以通过软连接创建它们:
            ln -s /home/data/Imagenet/val Assets/Imagenet_Valid
            ln -s /home/data/Imagenet/train Assets/Imagenet_Train
"""
from ppq.core.config import PPQ_CONFIG

# PPQ_CONFIG.PPQ_DEBUG=True

CFG_DEVICE = 'cuda'                            # 一个神奇的字符串，用来确定执行设备
CFG_BATCHSIZE = 16                             # 测试与calib时的 batchsize
CFG_INPUT_SHAPE = (CFG_BATCHSIZE, 3, 224, 224) # 用来确定模型输入的尺寸，好像 imagenet 都是这个尺寸
CFG_VALIDATION_DIR = '/data/val'   # 用来读取 validation dataset
CFG_TRAIN_DIR = '/data/train'        # 用来读取 train dataset，注意该集合将被用来 calibrate 你的模型
# CFG_PLATFORM = TargetPlatform.TRT_FP8     # 用来指定目标平台
CFG_PLATFORM = TargetPlatform.PPL_CUDA_INT8  
CFG_DUMP_PATH = 'Output/'                      # 所有模型保存的路径名
QUANT_SETTING = QuantizationSettingFactory.default_setting() # 用来指定量化配置

"""QAT setting"""
# ------------------------------------------------------------
# PPQ 提供基于 LSQ 的网络微调过程，这是推荐的做法
# 你将使用 Quant Setting 来调用微调过程，并调整微调参数
# ------------------------------------------------------------
QUANT_SETTING.lsq_optimization                            = True
QUANT_SETTING.lsq_optimization_setting.block_size         = 8
QUANT_SETTING.lsq_optimization_setting.steps              = 100
QUANT_SETTING.lsq_optimization_setting.lr                 = 1e-5
QUANT_SETTING.lsq_optimization_setting.gamma              = 0
QUANT_SETTING.lsq_optimization_setting.is_scale_trainable = True
QUANT_SETTING.lsq_optimization_setting.collecting_device  = 'cpu'

tp1_acc={}

with ENABLE_CUDA_KERNEL():
    if __name__ == '__main__':
        for model_builder, model_name in (
            # (torchvision.models.inception_v3, 'inception_v3'),

            # (torchvision.models.mnasnet0_5, 'mnasnet0_5'),
            # (torchvision.models.mnasnet1_0, 'mnasnet1_0'),

            # (torchvision.models.squeezenet1_0, 'squeezenet1_0'),
            # (torchvision.models.shufflenet_v2_x1_0, 'shufflenet_v2_x1_0'),
            # (torchvision.models.resnet18, 'resnet18'),

            # (torchvision.models.mobilenet.mobilenet_v2, 'mobilenet_v2'),
            # (torchvision.models.efficientnet.efficientnet_b0, 'efficientnet_b0'),

            # (torchvision.models.efficientnet.efficientnet_b1, 'efficientnet_b1'),
            # (torchvision.models.efficientnet.efficientnet_b2, 'efficientnet_b2'),
            (torchvision.models.efficientnet.efficientnet_b3, 'efficientnet_b3'),
            # (torchvision.models.efficientnet.efficientnet_b4, 'efficientnet_b4'),
            (torchvision.models.efficientnet.efficientnet_b5, 'efficientnet_b5'),
            # (torchvision.models.efficientnet.efficientnet_b6, 'efficientnet_b6'),
            (torchvision.models.efficientnet.efficientnet_b7, 'efficientnet_b7'),

            # (torchvision.models.resnet18, 'resnet18'),
            # (torchvision.models.resnet34, 'resnet34'),
            # (torchvision.models.resnet50, 'resnet50'),
            # (torchvision.models.resnet101, 'resnet101'),
            # (torchvision.models.resnet152, 'resnet152'),

            # (torchvision.models.mnasnet.mnasnet0_5, 'mnasnet0_5'),
            # (torchvision.models.mnasnet.mnasnet0_75, 'mnasnet0_75'),
            # (torchvision.models.mnasnet.mnasnet1_0, 'mnasnet1_0'),
            # (torchvision.models.mnasnet.mnasnet1_3, 'mnasnet1_3'),

            # (torchvision.models.mobilenet_v3_small, 'mobilenet_v3_small'),
            # (torchvision.models.efficientnet_v2_s, 'efficientnet_v2_s'),
            # (torchvision.models.efficientnet_v2_m, 'efficientnet_v2_m'),
            # (torchvision.models.efficientnet_v2_l, 'efficientnet_v2_l'),
            
        ):
            print(f'---------------------- PPQ Quantization Test Running with {model_name} ----------------------')
            model = model_builder(pretrained=True).to(CFG_DEVICE)
            print(model)

            # MyTQC = TQC(
            #     policy = QuantizationPolicy(
            #         QuantizationProperty.SYMMETRICAL + 
            #         QuantizationProperty.FLOATING +
            #         QuantizationProperty.PER_TENSOR + 
            #         QuantizationProperty.POWER_OF_2),
            #     rounding=RoundingPolicy.ROUND_HALF_EVEN,
            #     num_of_bits=8, quant_min=-448.0, quant_max=448.0, 
            #     exponent_bits=3, channel_axis=None,
            #     observer_algorithm='minmax'
            # )
            # quantizer = PFL.Quantizer(platform=TargetPlatform.TRT_FP8, graph=graph) # 取得 TRT_FP8 所对应的量化器

            """test"""
            dataloader = load_imagenet_from_directory(
                directory=CFG_TRAIN_DIR, batchsize=CFG_BATCHSIZE,
                shuffle=False, subset=5120, require_label=False,
                num_of_workers=8)

            ppq_quant_ir = quantize_torch_model(
                model=model, calib_dataloader=dataloader, input_shape=CFG_INPUT_SHAPE,
                calib_steps=5120 // CFG_BATCHSIZE, collate_fn=lambda x: x.to(CFG_DEVICE), verbose=1,
                device=CFG_DEVICE, platform=CFG_PLATFORM, setting=QUANT_SETTING)

            ppq_int8_report = evaluate_ppq_module_with_imagenet(
                model=ppq_quant_ir, imagenet_validation_dir=CFG_VALIDATION_DIR,
                batchsize=CFG_BATCHSIZE, device=CFG_DEVICE, verbose=True)

            # print(ppq_quant_ir.operations," =======operations========")
            # print(ppq_quant_ir.variables," =======variables========")
            # print(ppq_quant_ir.parameters()," =======parameters========")

            """analysis"""
            # reports = graphwise_error_analyse(
            #     graph=ppq_quant_ir, running_device=CFG_DEVICE, collate_fn=lambda x: x.to(CFG_DEVICE),
            #     dataloader=dataloader)
            # reports = layerwise_error_analyse(
            #     graph=ppq_quant_ir, running_device=CFG_DEVICE, collate_fn=lambda x: x.to(CFG_DEVICE), 
            #     dataloader=dataloader)  
            # report = statistical_analyse(
            #     graph=ppq_quant_ir, running_device=CFG_DEVICE, 
            #     collate_fn=lambda x: x.to(CFG_DEVICE), dataloader=dataloader)
            # from pandas import DataFrame
            # report = DataFrame(report)
            # report.to_csv('efficientnet_b0_int8.csv')

            # 从大到小排序单层误差
            # sensitivity = [(op_name, error) for op_name, error in reports.items()]
            # sensitivity = sorted(sensitivity, key=lambda x: x[1], reverse=True)

            # # 将前十个误差最大的层送上 FP32
            # for op_name, _ in sensitivity[: 10]:
            #     QUANT_SETTING.dispatching_table.append(operation=op_name, platform=TargetPlatform.FP32)
            # QUANT_SETTING.dispatching_table.append(operation='Conv_0',platform=TargetPlatform.FP32)
            # ppq_quant_ir = quantize_torch_model(
            #     model=model, calib_dataloader=dataloader, input_shape=CFG_INPUT_SHAPE,
            #     calib_steps=5120 // CFG_BATCHSIZE, collate_fn=lambda x: x.to(CFG_DEVICE), verbose=1,
            #     device=CFG_DEVICE, platform=CFG_PLATFORM, setting=QUANT_SETTING)

            # ppq_int8_report = evaluate_ppq_module_with_imagenet(
            #     model=ppq_quant_ir, imagenet_validation_dir=CFG_VALIDATION_DIR,
            #     batchsize=CFG_BATCHSIZE, device=CFG_DEVICE, verbose=True)


            """output"""
            # print(type(ppq_int8_report),ppq_int8_report)
            tp1_acc[model_name]=' * Prec@1 {top1:.3f} Prec@5 {top5:.3f}'.format(
            top1=sum(ppq_int8_report['top1_accuracy'])/len(ppq_int8_report['top1_accuracy']), 
            top5=sum(ppq_int8_report['top5_accuracy'])/len(ppq_int8_report['top5_accuracy']))
            print(tp1_acc[model_name])

            # export_ppq_graph(
            #     graph=ppq_quant_ir, 
            #     platform=TargetPlatform.ONNXRUNTIME,
            #     graph_save_to=f'{os.path.join(CFG_DUMP_PATH, model_name)}.onnx')
            
            # evaluate_onnx_module_with_imagenet(
            #     onnxruntime_model_path=f'{os.path.join(CFG_DUMP_PATH, model_name)}.onnx', 
            #     imagenet_validation_dir=CFG_VALIDATION_DIR, batchsize=CFG_BATCHSIZE, 
            #     device=CFG_DEVICE)
        print(tp1_acc)
    else:
        raise Exception('You may not import this file.')
