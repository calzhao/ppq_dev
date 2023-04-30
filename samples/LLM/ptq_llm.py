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
from accelerate import infer_auto_device_map

# PPQ_CONFIG.PPQ_DEBUG=True

CFG_DEVICE = 'cuda'                            # 一个神奇的字符串，用来确定执行设备
CFG_BATCHSIZE = 16                             # 测试与calib时的 batchsize
CFG_INPUT_SHAPE = (CFG_BATCHSIZE, 3, 224, 224) # 用来确定模型输入的尺寸，好像 imagenet 都是这个尺寸
CFG_VALIDATION_DIR = '/data/val'   # 用来读取 validation dataset
CFG_TRAIN_DIR = '/data/train'        # 用来读取 train dataset，注意该集合将被用来 calibrate 你的模型
CFG_PLATFORM = TargetPlatform.TRT_FP8     # 用来指定目标平台
# CFG_PLATFORM = TargetPlatform.PPL_CUDA_INT8  
CFG_DUMP_PATH = 'Output/'                      # 所有模型保存的路径名
# CACHE_DIR = ''
QUANT_SETTING = QuantizationSettingFactory.default_setting() # 用来指定量化配置
# device_map={'model.decoder.embed_tokens': 0, 'lm_head': 0, 'model.decoder.embed_positions': 0, 'model.decoder.final_layer_norm': 0, 'model.decoder.layers.0.self_attn': 0,
#  'model.decoder.layers.0.activation_fn': 0, 'model.decoder.layers.0.self_attn_layer_norm': 0, 'model.decoder.layers.0.fc1': 0, 'model.decoder.layers.0.fc2': 0, 'model.decoder.layers.0.final_layer_norm': 0, 
#  'model.decoder.layers.1': 1, 'model.decoder.layers.2': 1, 'model.decoder.layers.3': 1, 'model.decoder.layers.4': 1, 'model.decoder.layers.5': 1, 'model.decoder.layers.6': 1, 
#  'model.decoder.layers.7': 1, 'model.decoder.layers.8': 1, 'model.decoder.layers.9': 1, 'model.decoder.layers.10': 1, 'model.decoder.layers.11': 1, 'model.decoder.layers.12': 1, 
#  'model.decoder.layers.13': 1, 'model.decoder.layers.14': 1, 'model.decoder.layers.15': 1, 'model.decoder.layers.16': 1, 'model.decoder.layers.17': 1, 
#  'model.decoder.layers.18.self_attn': 1, 'model.decoder.layers.18.activation_fn': 1, 'model.decoder.layers.18.self_attn_layer_norm': 1,'model.decoder.layers.18.fc1': 2, 'model.decoder.layers.18.fc2': 2, 'model.decoder.layers.18.final_layer_norm': 2, 
#  'model.decoder.layers.19': 2, 'model.decoder.layers.20': 2, 'model.decoder.layers.21': 2, 'model.decoder.layers.22': 2, 'model.decoder.layers.23': 2, 'model.decoder.layers.24': 2, 
#  'model.decoder.layers.25': 2, 'model.decoder.layers.26': 2, 'model.decoder.layers.27': 2, 'model.decoder.layers.28': 2, 'model.decoder.layers.29': 2, 'model.decoder.layers.30': 2, 
#  'model.decoder.layers.31': 2, 'model.decoder.layers.32': 2, 'model.decoder.layers.33': 2, 'model.decoder.layers.34': 2, 'model.decoder.layers.35': 2, 'model.decoder.layers.36': 2, 
#  'model.decoder.layers.37': 2, 'model.decoder.layers.38': 2, 'model.decoder.layers.39': 2, 'model.decoder.layers.40': 2, 'model.decoder.layers.41': 2, 'model.decoder.layers.42': 2, 
#  'model.decoder.layers.43': 3, 'model.decoder.layers.44': 4, 'model.decoder.layers.45': 5, 'model.decoder.layers.46': 6, 'model.decoder.layers.47': 7}

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
    'facebook/opt-350m',
    # 'facebook/opt-1.3b',
    # 'facebook/opt-2.7b',
    # 'facebook/opt-6.7b',
    # 'facebook/opt-13b',
    # 'facebook/opt-30b',
    # 'facebook/opt-66b',

    # "decapoda-research/llama-7b-hf",
    # "decapoda-research/llama-13b-hf",

    # "/workspace/llama-7b-hf/",
    # "decapoda-research/llama-7b-hf",
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
        print(self.dataset[0])
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
            # print(torch.sum(last_token_logits),last_token_logits)
            pred = last_token_logits.argmax(dim=-1)
            # print(pred, label)
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
            # print(pred, label)
            total += label.size(0)
            hit += (pred == label).sum().item()
        acc = hit / total
        return acc

with ENABLE_CUDA_KERNEL():
    if __name__ == '__main__':

        dataset = load_dataset('lambada', split='validation')
        dataset = dataset.shuffle(seed=42).select(range(20))
        print(len(dataset),dataset[0])

        for model_checkpoint in model_list:
            # tokenizer = transformers.LlamaTokenizer.from_pretrained(model_checkpoint)
            # tokenizer.pad_token = "[PAD]"
            model_fp16 = AutoModelForCausalLM.from_pretrained(model_checkpoint, torch_dtype=torch.float32, device_map="auto") #.cuda()
            # model_fp16_cpu = AutoModelForCausalLM.from_pretrained(model_checkpoint, torch_dtype=torch.float32)
            print(model_fp16.hf_device_map)
            # model_fp16 = AutoModelForCausalLM.from_pretrained(model_checkpoint, torch_dtype=torch.float32).cuda()

            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)
            # device_map = infer_auto_device_map(model_fp16,max_memory = {0: "23GIB", 1: "23GIB", 2: "23GIB", 3: "23GIB", 4: "23GIB", 5: "23GIB", 6: "23GIB", 7: "23GIB"})
            # del model_fp16
            # print(device_map)
            # model_fp16 = AutoModelForCausalLM.from_pretrained(model_checkpoint, torch_dtype=torch.float32, device_map=device_map) #.cuda()
            # model_fp16 = AutoModelForCausalLM.from_pretrained(model_checkpoint, torch_dtype=torch.float32).cuda()
            # print(model_fp16)
            # print("model map: ",model_fp16.hf_device_map, model_fp16.dtype)
            # print("infer_auto_device_map: ",infer_auto_device_map(model_fp16))


            """Preprocessing the data"""
            # tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)
            # evaluator = Evaluator(dataset, tokenizer, CFG_DEVICE)
            evaluator = Evaluator(dataset, tokenizer, CFG_DEVICE)

            # tokenized_datasets = tokenized_datasets.remove_columns([sentence1_key])
            # if sentence2_key is not None:
            #     tokenized_datasets = tokenized_datasets.remove_columns([sentence2_key])
            # tokenized_datasets = tokenized_datasets.remove_columns(["idx"])
            # tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
            # tokenized_datasets.set_format("torch")
            # print("dataset len: ",len(tokenized_datasets["train"]),len(tokenized_datasets["validation_matched"]))
            # small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1600))
            # small_eval_dataset = tokenized_datasets["validation_matched"].shuffle(seed=42).select(range(8000))
            # train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=CFG_BATCHSIZE)
            # eval_dataloader = DataLoader(small_eval_dataset, batch_size=CFG_BATCHSIZE)

            """Eval the original model"""
            # acc_fp16 = evaluator.evaluate(model_fp16)
            # tp1_acc[model_checkpoint]=' * FP16 PREC {top1} '.format(top1=acc_fp16)
            # print(model_checkpoint,tp1_acc[model_checkpoint])

            """quantize"""
            for batch in evaluator.dataset:
                break
            # input_list = [k for k in batch if k!="labels" ]
            # collate_fn  = lambda x: {k:x[k].cuda() for k in input_list}
            input_ids = batch['input_ids'].to(CFG_DEVICE).unsqueeze(0)
            # model_fp16.eval()
            # torch.onnx.export(
            #     model=model_fp16, args=input_ids, 
            #     verbose=True, f='gpu.model', opset_version=11,
            # )
            model_fp16_cpu.eval()
            input_ids = input_ids.to('cpu')
            torch.onnx.export(
                model=model_fp16_cpu, args=input_ids, 
                verbose=True, f='cpu.model', opset_version=11,
            )
            break

            ppq_quant_ir = quantize_torch_model(
                model=model_fp16, calib_dataloader=evaluator.dataset.shuffle(seed=29).select(range(20)), input_shape=input_ids.shape, input_dtype=input_ids.dtype,
                # model=model_fp16, calib_dataloader=evaluator.dataset, input_shape=input_ids.shape, input_dtype=input_ids.dtype,
                calib_steps=20, collate_fn=lambda x: x['input_ids'].to(CFG_DEVICE).unsqueeze(0), verbose=1,
                device=CFG_DEVICE, platform=CFG_PLATFORM, setting=QUANT_SETTING, onnx_export_file="onnx.model")

            """evaluate"""
            executor = TorchExecutor(graph=ppq_quant_ir, device=CFG_DEVICE)
            model_forward_function = lambda input_tensor: torch.tensor(
                executor(*[input_tensor])[0])
            acc_fp8 = evaluator.evaluate_ppq(model_forward_function)
            tp1_acc[model_checkpoint]=' * FP16 PREC {top1} FP8 PREC {top5}'.format(top1=acc_fp16,top5=acc_fp8)
            # tp1_acc[model_checkpoint]=' * FP16 PREC {top1} FP8 PREC {top5}'.format(top1='hello',top5=acc_fp8)
            print(model_checkpoint,tp1_acc[model_checkpoint])

            """analysis"""
            # reports = graphwise_error_analyse(
            #     graph=ppq_quant_ir, running_device=CFG_DEVICE, collate_fn=lambda x: x['input_ids'].to(CFG_DEVICE).unsqueeze(0),
            #     dataloader=evaluator.dataset)
            # reports = layerwise_error_analyse(
            #     graph=ppq_quant_ir, running_device=CFG_DEVICE, collate_fn=lambda x: x['input_ids'].to(CFG_DEVICE).unsqueeze(0), 
            #     dataloader=evaluator.dataset)  
            # report = statistical_analyse(
            #     graph=ppq_quant_ir, running_device=CFG_DEVICE, 
            #     collate_fn=lambda x: x['input_ids'].to(CFG_DEVICE).unsqueeze(0), dataloader=evaluator.dataset)
            # from pandas import DataFrame
            # report = DataFrame(report)
            # report.to_csv(model_checkpoint[-4:]+'.csv')
            
            # np.save(model_checkpoint[-4:]+'_layer_int8_aligned',reports)
            # reports = np.load(model_checkpoint[-4:]+'_layer_int8_aligned.npy',allow_pickle=True)
            # reports = reports.item()

            # #从大到小排序单层误差
            # sensitivity = [(op_name, error) for op_name, error in reports.items()]
            # sensitivity = sorted(sensitivity, key=lambda x: x[1], reverse=True)

            # # 将前十个误差最大的层送上 FP32
            # for op_name, _ in sensitivity[: 20]:
            #     QUANT_SETTING.dispatching_table.append(operation=op_name, platform=TargetPlatform.FP32)
            # # QUANT_SETTING.dispatching_table.append(operation='Conv_0',platform=TargetPlatform.FP32)
            # ppq_quant_ir = quantize_torch_model(
            #     model=model_fp16, calib_dataloader=evaluator.dataset, input_shape=input_ids.shape, input_dtype=input_ids.dtype,
            #     calib_steps=100, collate_fn=lambda x: x['input_ids'].to(CFG_DEVICE).unsqueeze(0), verbose=1,
            #     device=CFG_DEVICE, platform=CFG_PLATFORM, setting=QUANT_SETTING)

            """evaluate"""
            # executor = TorchExecutor(graph=ppq_quant_ir, device=CFG_DEVICE)
            # model_forward_function = lambda input_tensor: torch.tensor(
            #     executor(*[input_tensor])[0])
            # acc_fp8 = evaluator.evaluate_ppq(model_forward_function)
            # tp1_acc[model_checkpoint]=' * AFTER SENT FP16 PREC {top1} FP8 PREC {top5}'.format(top1=acc_fp16,top5=acc_fp8)
            # print(model_checkpoint,tp1_acc[model_checkpoint])

        print(tp1_acc)
    else:
        raise Exception('You may not import this file.')
