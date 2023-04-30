import os
os.environ['TRANSFORMERS_CACHE'] = '/mnt/cache/huggingface/'
import torchvision
from ppq import *
from ppq.api import *
# from Utilities.Imagenet import (evaluate_mmlab_module_with_imagenet,
#                                 evaluate_onnx_module_with_imagenet,
#                                 evaluate_ppq_module_with_imagenet,
#                                 evaluate_torch_module_with_imagenet,
#                                 load_imagenet_from_directory)

"""
    使用这个脚本来测试量化 torchvision 中的典型分类模型
        使用 imagenet 中的数据测试量化精度与 calibration
        默认的 imagenet 数据集位置: Assets/Imagenet_Train, Assets/Imagenet_Valid
        你可以通过软连接创建它们:
            ln -s /home/data/Imagenet/val Assets/Imagenet_Valid
            ln -s /home/data/Imagenet/train Assets/Imagenet_Train
"""
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch
import gc
import argparse


CFG_DEVICE = 'cuda'                            # 一个神奇的字符串，用来确定执行设备
CFG_PLATFORM_FP8 = TargetPlatform.TRT_FP8     # 用来指定目标平台
CFG_PLATFORM_INT8 = TargetPlatform.PPL_CUDA_INT8  
CFG_PLATFORM_MIX = TargetPlatform.PPL_CUDA_MIX 
QUANT_SETTING = QuantizationSettingFactory.default_setting() # 用来指定量化配置

"""QAT setting"""
# ------------------------------------------------------------
# PPQ 提供基于 LSQ 的网络微调过程，这是推荐的做法
# 你将使用 Quant Setting 来调用微调过程，并调整微调参数
# ------------------------------------------------------------
# QUANT_SETTING.lsq_optimization                            = True
# QUANT_SETTING.lsq_optimization_setting.block_size         = 8
# QUANT_SETTING.lsq_optimization_setting.steps              = 100
# QUANT_SETTING.lsq_optimization_setting.lr                 = 1e-5
# QUANT_SETTING.lsq_optimization_setting.gamma              = 0
# QUANT_SETTING.lsq_optimization_setting.is_scale_trainable = True
# QUANT_SETTING.lsq_optimization_setting.collecting_device  = 'cpu'

model_list=[
    'facebook/opt-125m',
    'facebook/opt-350m',
    'facebook/opt-1.3b',
    'facebook/opt-2.7b',
    'facebook/opt-6.7b',
    'facebook/opt-13b',
    'facebook/opt-30b',
    'facebook/opt-66b',

    "decapoda-research/llama-7b-hf",
    "decapoda-research/llama-13b-hf",
    "decapoda-research/llama-30b-hf",
    "decapoda-research/llama-65b-hf",
]
# seq = ["input_ids", "attention_mask", "token_type_ids", 
#         "position_ids", "head_mask", "inputs_embeds", 
#         "labels","output_attentions","output_hidden_states"]
tp1_acc={}
TASK_TO_EVALUATOR = {
    "wikitext": WikiEvaluator,
    "lambada": LambadaEvaluator,
    "piqa": PiqaEvaluator,
    "hellaswag": HellaEvaluator,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=model_list)
    parser.add_argument("--task", required=True, choices=["lambada","hellaswag","piqa","wikitext"])
    parser.add_argument("--model_args", default="")
    parser.add_argument("--output_path", default="out.txt")
    parser.add_argument("--calib_steps", default=100)
    parser.add_argument("--layerwise", default='perlayer', choices=["perlayer","final"])
    parser.add_argument("--method", default='snr', choices=["snr","cosine","mse"])

    return parser.parse_args()

with ENABLE_CUDA_KERNEL():
    if __name__ == '__main__':

        print("before program")
        print("memory_allocated",torch.cuda.memory_allocated())
        print("memory_catched",torch.cuda.memory_reserved())

        args = parse_args()
        CALIB_STEP = args.calib_steps

        if args.layerwise=='perlayer':
            analysis_func = layerwise_error_analyse_v2
        else:
            analysis_func = layerwise_error_analyse

        model_checkpoint = args.model
        """Preprocessing the data"""
        print(model_checkpoint)
        model_fp16 = AutoModelForCausalLM.from_pretrained(model_checkpoint, torch_dtype=torch.float32, device_map="auto")
        print(model_fp16.hf_device_map)
        # model_fp16 = AutoModelForCausalLM.from_pretrained(model_checkpoint, torch_dtype=torch.float32).cuda()

        if "llama" in model_checkpoint:
            tokenizer = transformers.LlamaTokenizer.from_pretrained(model_checkpoint, use_fast=False)
            tokenizer.pad_token = "[PAD]"
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)

        if args.task == "wikitext":
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1',split='validation')
        else:
            dataset = load_dataset(args.task, split='validation')
        print(len(dataset),dataset[0])

        evaluator = TASK_TO_EVALUATOR[args.task](dataset, tokenizer, CFG_DEVICE, model_fp16, CALIB_STEP)

        """Eval the original model"""
        acc_fp16 = evaluator.evaluate(model_fp16)
        tp1_acc[model_checkpoint]=' * FP16 PREC {top1} '.format(top1=acc_fp16)
        print(model_checkpoint,tp1_acc[model_checkpoint])

        batch = evaluator.sample_batch()
        input_ids = batch.to(CFG_DEVICE)

        """quantize int"""
        if os.path.exists(model_checkpoint[-6:]+'_layer_int8_'+ args.task + args.method +'_v2.npy'):
            reports_int8 = np.load(model_checkpoint[-6:]+'_layer_int8_'+ args.task + args.method +'_v2.npy',allow_pickle=True)
            reports_int8 = reports_int8.item()
        else:
            ppq_quant_ir_INT8 = quantize_torch_model(
                model=model_fp16, calib_dataloader=evaluator.calib_dataloader, input_shape=input_ids.shape, input_dtype=input_ids.dtype,
                # model=model_fp16, calib_dataloader=evaluator.dataset, input_shape=input_ids.shape, input_dtype=input_ids.dtype,
                calib_steps=CALIB_STEP, collate_fn=evaluator.my_collate_fn, verbose=1,
                device=CFG_DEVICE, platform=CFG_PLATFORM_INT8, setting=QUANT_SETTING)
            reports_int8 = analysis_func(
                graph=ppq_quant_ir_INT8, running_device=CFG_DEVICE, collate_fn=evaluator.my_collate_fn, 
                dataloader=evaluator.dataset, method=args.method)
            np.save(model_checkpoint[-6:]+'_layer_int8_'+ args.task + args.method +'_v2.npy',reports_int8)
            """evaluate"""
            executor = TorchExecutor(graph=ppq_quant_ir_INT8, device=CFG_DEVICE)
            model_forward_function = lambda input_tensor: torch.tensor(
                executor(*[input_tensor])[0])
            acc_int8 = evaluator.evaluate_ppq(model_forward_function)
            tp1_acc[model_checkpoint]=' *  INT8 PREC {top5}'.format(top5=acc_int8)
            # tp1_acc[model_checkpoint]=' * INT8 PREC {top1} FP8 PREC {top3} MIX PREC {top5}'.format(top1=acc_int8, top3=acc_fp8 ,top5=acc_mix8)
            print(model_checkpoint,tp1_acc[model_checkpoint])

            del model_forward_function
            del ppq_quant_ir_INT8
            del executor
            gc.collect()
            torch.cuda.empty_cache()

        print("mid program")
        print("memory_allocated",torch.cuda.memory_allocated())
        print("memory_catched",torch.cuda.memory_reserved())

        """quantize fp"""
        if os.path.exists(model_checkpoint[-6:]+'_layer_fp8_'+ args.task + args.method +'_v2.npy'):
            reports_fp8 = np.load(model_checkpoint[-6:]+'_layer_fp8_'+ args.task + args.method +'_v2.npy',allow_pickle=True)
            reports_fp8 = reports_fp8.item()
        else:
            ppq_quant_ir_FP8 = quantize_torch_model(
                model=model_fp16, calib_dataloader=evaluator.calib_dataloader, input_shape=input_ids.shape, input_dtype=input_ids.dtype,
                # model=model_fp16, calib_dataloader=evaluator.dataset, input_shape=input_ids.shape, input_dtype=input_ids.dtype,
                calib_steps=CALIB_STEP, collate_fn=evaluator.my_collate_fn, verbose=1,
                device=CFG_DEVICE, platform=CFG_PLATFORM_FP8, setting=QUANT_SETTING)
            reports_fp8 = analysis_func(
                graph=ppq_quant_ir_FP8, running_device=CFG_DEVICE, collate_fn=evaluator.my_collate_fn,
                dataloader=evaluator.dataset, method=args.method) 
            np.save(model_checkpoint[-6:]+'_layer_fp8_'+ args.task + args.method +'_v2.npy',reports_fp8)  
            """evaluate"""
            executor = TorchExecutor(graph=ppq_quant_ir_FP8, device=CFG_DEVICE)
            model_forward_function = lambda input_tensor: torch.tensor(
                executor(*[input_tensor])[0])
            acc_fp8 = evaluator.evaluate_ppq(model_forward_function)
            tp1_acc[model_checkpoint]=' *  FP8 PREC {top5}'.format(top5=acc_fp8)
            # tp1_acc[model_checkpoint]=' * INT8 PREC {top1} FP8 PREC {top3} MIX PREC {top5}'.format(top1=acc_int8, top3=acc_fp8 ,top5=acc_mix8)
            print(model_checkpoint,tp1_acc[model_checkpoint])

            del model_forward_function
            del ppq_quant_ir_FP8
            del executor
            gc.collect()
            torch.cuda.empty_cache()

        """set the final model"""
        op_cnt, fp_cnt= 0,0
        for op_name, _ in reports_int8.items():
            if op_name not in reports_fp8:
                continue
            op_cnt += 1
            # print(op_name, reports_int8[op_name]<reports_fp8[op_name])
            if reports_int8[op_name]>reports_fp8[op_name]:
                QUANT_SETTING.dispatching_table.append(operation=op_name, platform=TargetPlatform.TRT_FP8)
                fp_cnt += 1
        tp1_acc[model_checkpoint]=' * op cnt {top1} fp cnt {top5} fp percent {pct}'.format(
            top1=op_cnt, top5=fp_cnt, pct=fp_cnt/op_cnt)
        print(model_checkpoint,tp1_acc[model_checkpoint])
        if args.output_path:
            with open(args.output_path, "a") as f:
                f.write(model_checkpoint+tp1_acc[model_checkpoint]+"  ")

        ppq_quant_ir = quantize_torch_model( model=model_fp16, calib_dataloader=evaluator.calib_dataloader, 
            input_shape=input_ids.shape, input_dtype=input_ids.dtype, calib_steps=CALIB_STEP, 
            collate_fn=evaluator.my_collate_fn, verbose=1, device=CFG_DEVICE, platform=CFG_PLATFORM_MIX, 
            setting=QUANT_SETTING)

        """evaluate"""
        executor = TorchExecutor(graph=ppq_quant_ir, device=CFG_DEVICE)
        model_forward_function = lambda input_tensor: torch.tensor(
            executor(*[input_tensor])[0])
        acc_mix8 = evaluator.evaluate_ppq(model_forward_function)
        tp1_acc[model_checkpoint]=' *  MIX PREC {top5}'.format(top5=acc_mix8)
        print(model_checkpoint,tp1_acc[model_checkpoint])

        try:
            tp1_acc[model_checkpoint]=' FP16 PREC {top0} * INT8 PREC {top1} FP8 PREC {top3} MIX PREC {top5}'.format(
                top0=acc_fp16, top1=acc_int8, top3=acc_fp8 ,top5=acc_mix8)
            print(model_checkpoint,tp1_acc[model_checkpoint])
        except Exception:
            tp1_acc[model_checkpoint]=' FP16 PREC {top0} * MIX PREC {top5}'.format(
                top0=acc_fp16, top5=acc_mix8)
            print(model_checkpoint,tp1_acc[model_checkpoint])

        if args.output_path:
            with open(args.output_path, "a") as f:
                f.write(model_checkpoint+" ")
                f.write(tp1_acc[model_checkpoint]+" \n")
        
        del model_forward_function
        del ppq_quant_ir
        del executor
        gc.collect()
        torch.cuda.empty_cache()

        del model_fp16
        del evaluator
        del tokenizer
        del input_ids
        gc.collect()
        torch.cuda.empty_cache()
        print("after program")
        print("memory_allocated",torch.cuda.memory_allocated())
        print("memory_catched",torch.cuda.memory_reserved())

    else:
        raise Exception('You may not import this file.')
