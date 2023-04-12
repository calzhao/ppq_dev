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
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import re
import numpy as np


# PPQ_CONFIG.PPQ_DEBUG=True

CFG_DEVICE = 'cuda'                            # 一个神奇的字符串，用来确定执行设备
CFG_BATCHSIZE = 16                             # 测试与calib时的 batchsize
CFG_INPUT_SHAPE = (CFG_BATCHSIZE, 3, 224, 224) # 用来确定模型输入的尺寸，好像 imagenet 都是这个尺寸
CFG_VALIDATION_DIR = '/data/val'   # 用来读取 validation dataset
CFG_TRAIN_DIR = '/data/train'        # 用来读取 train dataset，注意该集合将被用来 calibrate 你的模型
CFG_PLATFORM = TargetPlatform.TRT_FP8     # 用来指定目标平台
# CFG_PLATFORM = TargetPlatform.PPL_CUDA_INT8  
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
    # 'facebook/opt-6.7b',
    'facebook/opt-13b',
    'facebook/opt-30b',
    # 'facebook/opt-66b',
]
# seq = ["input_ids", "attention_mask", "token_type_ids", 
#         "position_ids", "head_mask", "inputs_embeds", 
#         "labels","output_attentions","output_hidden_states"]
tp1_acc={}

class Evaluator:
    def __init__(self, dataset, tokenizer, device, _model_call):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self._model_call=_model_call
        
        def tokenize_function(examples):
            out_doc = self._process_doc(examples)
            out_doc['context_enc'] = self.tokenizer(self.doc_to_text(out_doc),truncation=True)
            out_doc['continuation_enc'] = self.tokenizer(self.doc_to_target(out_doc),truncation=True)
            return out_doc
        
        self.dataset = self.dataset.map(tokenize_function, batched=False)
        # self.dataset.set_format(type='torch', columns=['input_ids','attention_mask','context_enc','continuation_enc'])

        def set_padding(examples):
            # print(examples)
            context_enc = max(len(elem['input_ids']) for elem in examples['context_enc'])
            # print([max(map(len, ele['input_ids'])) for ele in examples['continuation_enc']])
            continuation_enc = max([max(map(len, ele['input_ids'])) for ele in examples['continuation_enc'] ])
            self.padding_length = context_enc+continuation_enc+20
            print(context_enc,continuation_enc)
            return None
        self.dataset.map(set_padding, batched=True)

        print(self.padding_length, self.dataset[0])

    def _process_doc(self, doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": self.preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [" "+self.preprocess(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }
        return out_doc

    def preprocess(self, text):
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    def doc_to_text(self, doc):
        return doc["query"]

    def doc_to_target(self, doc):
        # print("target1 ", " " + doc["choices"][0])
        return doc["choices"]

    def _loglikelihood_tokens(self, context_enc, continuation_enc, foward_func=None, sample=False):

        padding_length = self.padding_length

        # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
        # tensors, then we pack them together into a batch, call the model, and then pick it all apart
        # again because vectorizing is annoying

        # sanity check
        # assert len(context_enc) > 0
        # assert len(continuation_enc) > 0

        # print("enc shape ",len(context_enc),len(continuation_enc))
        # print(context_enc,continuation_enc)
        # how this all works:
        #          CTX      CONT
        # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
        # gpt2    \               \
        # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
        # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

        # when too long to fit in context, truncate from the left
        inp = torch.tensor(
            (context_enc + continuation_enc)[:][:-1],
            dtype=torch.long,
        ).to(self.device)
        (inplen,) = inp.shape

        # print("inp.shape",inp.shape)

        # since in _collate we make sure length is descending, the longest is always the first one.
        padding_length = (
            padding_length if padding_length is not None else inplen
        )

        # pad length from seq to padding_length
        inp = torch.cat(
            [
                inp,  # [seq]
                torch.zeros(padding_length - inplen, dtype=torch.long).to(
                    inp.device
                ),  # [padding_length - seq]
            ],
            dim=0,
        )
        inp = inp.unsqueeze(0)  # [1, padding_length]
        cont_toks = continuation_enc

        if sample:
            return inp
        if foward_func == None:
            output = self._model_call(inp)
            logits = F.log_softmax(
                output.logits, dim=-1
            ).cpu()  # [batch, padding_length, vocab]
        else:
            output = foward_func(inp)
            logits = F.log_softmax(
                output, dim=-1
            ).cpu()  # [batch, padding_length, vocab]
        # print("logits",output.logits.shape)

        # Slice to original seq length
        contlen = len(cont_toks)
        # print(inplen,contlen)
        # print(logits.shape,logits[:,inplen - contlen : inplen].shape)
        logits = logits[:,inplen - contlen : inplen]  # [1, seq, vocab]

        # Check if per-token argmax is exactly equal to continuation
        greedy_tokens = logits.argmax(dim=-1)
        # print("greedy_tokens.shape",greedy_tokens.shape)
        cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(
            0
        )  # [1, seq]
        max_equal = (greedy_tokens == cont_toks).all()

        # Obtain log-probs at the corresponding continuation token indices
        # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
        logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
            -1
        )  # [1, seq]
        # print("logits.shape",logits.shape)

        # Answer: (log prob, is-exact-match)
        answer = (float(logits.sum()), bool(max_equal))
        # print("answer",answer)

        return answer
    
    @torch.no_grad()
    def sample_batch(self):
        return torch.ones((1,self.padding_length),dtype=torch.int32)

    def my_collate_fn(self, batch):
        context_enc = batch['context_enc']['input_ids']
        continuation_enc1 = batch['continuation_enc']['input_ids'][0]
        input1 = self._loglikelihood_tokens(context_enc,continuation_enc1,sample=True)
        return input1.to(self.device)

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in self.dataset:
            context_enc = batch['context_enc']['input_ids']
            label = batch['gold']
            outputs = []
            for continuation_enc in batch['continuation_enc']['input_ids']:
                output = self._loglikelihood_tokens(context_enc,continuation_enc)[0]
                outputs.append(output)
                # print(torch.sum(last_token_logits),last_token_logits)
            pred = np.argmax(outputs)
            # print(pred, label)
            total += 1
            hit += pred == label
        acc = hit / total
        return acc
    
    def evaluate_ppq(self, fw_func):
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in self.dataset:

            context_enc = batch['context_enc']['input_ids']
            label = batch['gold']
            outputs = []
            for continuation_enc in batch['continuation_enc']['input_ids']:
                output = self._loglikelihood_tokens(context_enc,continuation_enc,foward_func=fw_func)[0]
                outputs.append(output)
                # print(torch.sum(last_token_logits),last_token_logits)
            pred = np.argmax(outputs)
            # print(pred, label)
            total += 1
            hit += pred == label

        acc = hit / total
        return acc

with ENABLE_CUDA_KERNEL():
    if __name__ == '__main__':

        dataset = load_dataset('hellaswag', split='validation')
        dataset = dataset.shuffle(seed=42).select(range(10000))
        print(len(dataset),dataset[0])


        for model_checkpoint in model_list:
            model_fp16 = AutoModelForCausalLM.from_pretrained(model_checkpoint, torch_dtype=torch.float32, device_map="auto") #.cuda()
            # model_fp16 = AutoModelForCausalLM.from_pretrained(model_checkpoint, torch_dtype=torch.float32).cuda()
            print(model_fp16.hf_device_map,model_fp16.dtype)

            """Preprocessing the data"""
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)
            evaluator = Evaluator(dataset, tokenizer, CFG_DEVICE, model_fp16)

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
            acc_fp16 = evaluator.evaluate(model_fp16)
            tp1_acc[model_checkpoint]=' * FP16 PREC {top1} '.format(top1=acc_fp16)
            print(model_checkpoint,tp1_acc[model_checkpoint])

            # """quantize"""
            batch = evaluator.sample_batch()
            # input_list = [k for k in batch if k!="labels" ]
            # collate_fn  = lambda x: {k:x[k].cuda() for k in input_list}
            input_ids = batch.to(CFG_DEVICE)
            ppq_quant_ir = quantize_torch_model(
                model=model_fp16, calib_dataloader=evaluator.dataset.shuffle(seed=29).select(range(100)), input_shape=input_ids.shape, input_dtype=input_ids.dtype,
                # model=model_fp16, calib_dataloader=evaluator.dataset, input_shape=input_ids.shape, input_dtype=input_ids.dtype,
                calib_steps=100, collate_fn=evaluator.my_collate_fn, verbose=1,
                device=CFG_DEVICE, platform=CFG_PLATFORM, setting=QUANT_SETTING)

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
