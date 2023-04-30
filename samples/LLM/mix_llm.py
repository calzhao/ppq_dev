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
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
import torch
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from transformers import GPT2Tokenizer
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc

# PPQ_CONFIG.PPQ_DEBUG=True

CFG_DEVICE = 'cuda'                            # 一个神奇的字符串，用来确定执行设备
CFG_BATCHSIZE = 16                             # 测试与calib时的 batchsize
CFG_INPUT_SHAPE = (CFG_BATCHSIZE, 3, 224, 224) # 用来确定模型输入的尺寸，好像 imagenet 都是这个尺寸
CFG_VALIDATION_DIR = '/data/val'   # 用来读取 validation dataset
CFG_TRAIN_DIR = '/data/train'        # 用来读取 train dataset，注意该集合将被用来 calibrate 你的模型
CFG_PLATFORM_FP8 = TargetPlatform.TRT_FP8     # 用来指定目标平台
CFG_PLATFORM_INT8 = TargetPlatform.PPL_CUDA_INT8  
CFG_PLATFORM_MIX = TargetPlatform.PPL_CUDA_MIX 
CFG_DUMP_PATH = 'Output/'                      # 所有模型保存的路径名
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
    # 'facebook/opt-125m',
    # 'facebook/opt-350m',
    # 'facebook/opt-1.3b',
    # 'facebook/opt-2.7b',
    'facebook/opt-6.7b',
    # 'facebook/opt-13b',
    # 'facebook/opt-30b',
    # 'facebook/opt-66b',

    # "decapoda-research/llama-7b-hf",
    # "decapoda-research/llama-13b-hf",
    # "decapoda-research/llama-30b-hf",
    # "decapoda-research/llama-65b-hf",
]
# seq = ["input_ids", "attention_mask", "token_type_ids", 
#         "position_ids", "head_mask", "inputs_embeds", 
#         "labels","output_attentions","output_hidden_states"]
tp1_acc={}
class Evaluator:
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'],padding='longest',truncation=True)
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        # print(self.dataset[0])
        self.dataset.set_format(type='torch', columns=['input_ids','attention_mask'])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in self.dataset:
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)
            attention_mask = batch['attention_mask'].to(self.device).unsqueeze(0)
            # label = input_ids[:, -1]
            label = input_ids[:,int(torch.sum(batch['attention_mask'])-1)]
            # outputs = model(input_ids,attention_mask=attention_mask)
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, int(torch.sum(batch['attention_mask'])-2), :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        acc = hit / total
        return acc
    
    def evaluate_ppq(self, fw_func):
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in self.dataset:
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)
            # label = input_ids[:, -1]
            label = input_ids[:,int(torch.sum(batch['attention_mask'])-1)]
            outputs = fw_func(input_ids)
            # print(outputs.shape)
            last_token_logits = outputs[:, int(torch.sum(batch['attention_mask'])-2), :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        acc = hit / total
        return acc

with ENABLE_CUDA_KERNEL():
    if __name__ == '__main__':

        dataset = load_dataset('lambada', split='validation')
        dataset = dataset.shuffle(seed=42).select(range(1000))
        print(len(dataset),dataset[0])

        for model_checkpoint in model_list:
            # model_fp16 = AutoModelForCausalLM.from_pretrained(model_checkpoint, torch_dtype=torch.float32, device_map="auto") 

            """Preprocessing the data"""
            # tokenizer = transformers.LlamaTokenizer.from_pretrained(model_checkpoint)
            # tokenizer.pad_token = "[PAD]"
            # model_fp16 = AutoModelForCausalLM.from_pretrained(model_checkpoint, torch_dtype=torch.float32, device_map="auto") #.cuda()
            model_fp16 = AutoModelForCausalLM.from_pretrained(model_checkpoint, torch_dtype=torch.float32).cuda()
            # print(model_fp16.hf_device_map)
            # model_fp16 = AutoModelForCausalLM.from_pretrained(model_checkpoint, torch_dtype=torch.float32).cuda()

            if "llama" in model_checkpoint:
                tokenizer = transformers.LlamaTokenizer.from_pretrained(model_checkpoint, use_fast=False)
                tokenizer.pad_token = "[PAD]"
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)
            evaluator = Evaluator(dataset, tokenizer, CFG_DEVICE)

            # """Eval the original model"""
            # acc_fp16 = evaluator.evaluate(model_fp16)
            # tp1_acc[model_checkpoint]=' * FP16 PREC {top1} '.format(top1=acc_fp16)
            # print(model_checkpoint,tp1_acc[model_checkpoint])

            for batch in evaluator.dataset:
                break
            input_ids = batch['input_ids'].to(CFG_DEVICE).unsqueeze(0)

            """quantize int"""
            if os.path.exists(model_checkpoint[-6:]+'_layer_int8_aligned_v2.npy') and False:
                reports_int8 = np.load(model_checkpoint[-6:]+'_layer_int8_aligned_v2.npy',allow_pickle=True)
                reports_int8 = reports_int8.item()
            else:
                ppq_quant_ir_INT8 = quantize_torch_model(
                    model=model_fp16, calib_dataloader=evaluator.dataset.shuffle(seed=29).select(range(100)), input_shape=input_ids.shape, input_dtype=input_ids.dtype,
                    # model=model_fp16, calib_dataloader=evaluator.dataset, input_shape=input_ids.shape, input_dtype=input_ids.dtype,
                    calib_steps=100, collate_fn=lambda x: x['input_ids'].to(CFG_DEVICE).unsqueeze(0), verbose=1,
                    device=CFG_DEVICE, platform=CFG_PLATFORM_INT8, setting=QUANT_SETTING)
                reports_int8 = layerwise_error_analyse_v2(
                    graph=ppq_quant_ir_INT8, running_device=CFG_DEVICE, collate_fn=lambda x: x['input_ids'].to(CFG_DEVICE).unsqueeze(0), 
                    dataloader=evaluator.dataset, method='mse')
                np.save(model_checkpoint[-6:]+'_layer_int8_aligned_v2',reports_int8)
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
 
            """quantize fp"""
            if os.path.exists(model_checkpoint[-6:]+'_layer_fp8_v2.npy') and False:
                reports_fp8 = np.load(model_checkpoint[-6:]+'_layer_fp8_v2.npy',allow_pickle=True)
                reports_fp8 = reports_fp8.item()
            else:
                ppq_quant_ir_FP8 = quantize_torch_model(
                    model=model_fp16, calib_dataloader=evaluator.dataset.shuffle(seed=29).select(range(100)), input_shape=input_ids.shape, input_dtype=input_ids.dtype,
                    # model=model_fp16, calib_dataloader=evaluator.dataset, input_shape=input_ids.shape, input_dtype=input_ids.dtype,
                    calib_steps=100, collate_fn=lambda x: x['input_ids'].to(CFG_DEVICE).unsqueeze(0), verbose=1,
                    device=CFG_DEVICE, platform=CFG_PLATFORM_FP8, setting=QUANT_SETTING)
                reports_fp8 = layerwise_error_analyse_v2(
                    graph=ppq_quant_ir_FP8, running_device=CFG_DEVICE, collate_fn=lambda x: x['input_ids'].to(CFG_DEVICE).unsqueeze(0), 
                    dataloader=evaluator.dataset, method='mse') 
                np.save(model_checkpoint[-6:]+'_layer_fp8_v2',reports_fp8)  
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

            # """analysis"""
            # # reports_int8 = layerwise_error_analyse(
            # #     graph=ppq_quant_ir_INT8, running_device=CFG_DEVICE, collate_fn=lambda x: x['input_ids'].to(CFG_DEVICE).unsqueeze(0), 
            # #     dataloader=evaluator.dataset)
            # # reports_fp8 = layerwise_error_analyse(
            # #     graph=ppq_quant_ir_FP8, running_device=CFG_DEVICE, collate_fn=lambda x: x['input_ids'].to(CFG_DEVICE).unsqueeze(0), 
            # #     dataloader=evaluator.dataset)    
            
            # # np.save(model_checkpoint[-4:]+'_layer_int8_aligned',reports_int8)
            # # np.save(model_checkpoint[-4:]+'_layer_fp8',reports_fp8)
            # reports_int8 = np.load(model_checkpoint[-4:]+'_layer_int8_aligned.npy',allow_pickle=True)
            # reports_fp8 = np.load(model_checkpoint[-4:]+'_layer_fp8.npy',allow_pickle=True)
            # reports_int8 = reports_int8.item()
            # reports_fp8 = reports_fp8.item()

            """set the final model"""
            #从大到小排序单层误差
            # sensitivity = [(op_name, error) for op_name, error in reports_int8.items()]
            # sensitivity = sorted(sensitivity, key=lambda x: x[1], reverse=True)
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

            # 将前十个误差最大的层送上 FP32
            # for op_name, _ in sensitivity[: 5]:
            #     print(op_name)
            #     QUANT_SETTING.dispatching_table.append(operation=op_name, platform=TargetPlatform.TRT_FP8)

            ppq_quant_ir = quantize_torch_model(
                model=model_fp16, calib_dataloader=evaluator.dataset.shuffle(seed=29).select(range(100)), input_shape=input_ids.shape, input_dtype=input_ids.dtype,
                calib_steps=100, collate_fn=lambda x: x['input_ids'].to(CFG_DEVICE).unsqueeze(0), verbose=1,
                device=CFG_DEVICE, platform=CFG_PLATFORM_MIX, setting=QUANT_SETTING)

            """evaluate"""
            executor = TorchExecutor(graph=ppq_quant_ir, device=CFG_DEVICE)
            model_forward_function = lambda input_tensor: torch.tensor(
                executor(*[input_tensor])[0])
            acc_mix8 = evaluator.evaluate_ppq(model_forward_function)
            tp1_acc[model_checkpoint]=' *  MIX PREC {top5}'.format(top5=acc_mix8)
            print(model_checkpoint,tp1_acc[model_checkpoint])

            tp1_acc[model_checkpoint]=' * INT8 PREC {top1} FP8 PREC {top3} MIX PREC {top5}'.format(top1=acc_int8, top3=acc_fp8 ,top5=acc_mix8)
            print(model_checkpoint,tp1_acc[model_checkpoint])
            
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
            # export_ppq_graph(
            #     graph=ppq_quant_ir, 
            #     platform=TargetPlatform.ONNXRUNTIME,
            #     graph_save_to=f'{os.path.join(CFG_DUMP_PATH, model_name)}.onnx')

        print(tp1_acc)
    else:
        raise Exception('You may not import this file.')
