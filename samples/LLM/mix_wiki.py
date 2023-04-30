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
from lm_eval import utils
from lm_eval.metrics import mean, weighted_perplexity, weighted_mean, bits_per_byte
import re
import random


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
CALIB_STEP = 100

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
    # 'facebook/opt-13b',
    # 'facebook/opt-30b',
    # 'facebook/opt-66b',

    'facebook/opt-6.7b',
    'facebook/opt-2.7b',
    # 'facebook/opt-1.3b',
    # 'facebook/opt-350m',
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
        self.eot_token_id = self.tokenizer.eos_token_id

        trainenc = tokenizer("\n\n".join(dataset['text']))
        print("nn seq",tokenizer("\n\n"))
        print("text length",len(trainenc['input_ids']))

        random.seed(0)
        trainloader = []
        self.seqlen=_model_call.config.max_position_embeddings
        nsamples = 128
        for _ in range(nsamples):
            i = random.randint(0, len(trainenc.input_ids) - self.seqlen - 1)
            j = i + self.seqlen
            inp = trainenc.input_ids[i:j]
            trainloader.append(inp)
        # return trainloader, testenc
        self.dataset = trainloader
        print("dataset len",len(self.dataset))

        self.calib_dataloader = random.sample(trainloader, CALIB_STEP)

        self.padding_length=self.seqlen
        print("padding_length",self.padding_length)


    @property
    def max_length(self):
        try:
            return self._model_call.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self._model_call.config.max_position_embeddings

    @torch.no_grad()
    def loglikelihood_rolling(self, token, foward_func=None,sample=False):
        # TODO: Implement caching once we've confirmed the perplexity implementation
        # TODO: automatic batch size detection for vectorization

        rolling_token_windows = list(
            map(
                utils.make_disjoint_window,
                utils.get_rolling_token_windows(
                    token_list=token,
                    prefix_token=self.eot_token_id,
                    max_seq_len=self.max_length,
                    context_len=1,
                ),
            )
        )
        string_nll = self._loglikelihood_tokens( *rolling_token_windows[0],foward_func=foward_func, sample=sample)
        if sample:
            return string_nll

        string_nll = string_nll[0]

        return string_nll

    @torch.no_grad()
    def _loglikelihood_tokens(self, context_enc, continuation_enc, foward_func=None, sample=False):

        padding_length = self.padding_length

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

    @torch.no_grad()
    def my_collate_fn(self, batch):
        context_enc = batch
        input1 = self.loglikelihood_rolling(context_enc,sample=True)
        return input1.to(self.device)

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        outputs = []
        for batch in self.dataset:
            # context_enc = batch['enc']['input_ids']
            # print(batch)
            # words = self.count_words(batch['raw'])
            context_enc = batch
            # print("context_enc",context_enc)
            words = len(batch)
            loglikelihood = self.loglikelihood_rolling(context_enc)
            # print(loglikelihood,words)
            outputs.append((loglikelihood, words))
        # acc = hit / total
        return weighted_perplexity(outputs)

    @torch.no_grad()
    def evaluate_ppq(self, fw_func):
        # The task is to predict the last word of the input.
        outputs = []
        for batch in self.dataset:
            context_enc = batch
            words = len(batch)
            loglikelihood = self.loglikelihood_rolling(context_enc,foward_func=fw_func)
            outputs.append((loglikelihood, words))
        # acc = hit / total
        return weighted_perplexity(outputs)

with ENABLE_CUDA_KERNEL():
    if __name__ == '__main__':

        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1',split='validation')
        # dataset = dataset.shuffle(seed=42).select(range(1000))
        print(len(dataset),dataset[0])


        for model_checkpoint in model_list:
            model_fp16 = AutoModelForCausalLM.from_pretrained(model_checkpoint, torch_dtype=torch.float32, device_map="auto") #.cuda()
            print(model_fp16.hf_device_map)
            # model_fp16 = AutoModelForCausalLM.from_pretrained(model_checkpoint, torch_dtype=torch.float32).cuda()

            if "llama" in model_checkpoint:
                tokenizer = transformers.LlamaTokenizer.from_pretrained(model_checkpoint, use_fast=False)
                tokenizer.pad_token = "[PAD]"
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)
            evaluator = Evaluator(dataset, tokenizer, CFG_DEVICE, model_fp16)

            """Eval the original model"""
            acc_fp16 = evaluator.evaluate(model_fp16)
            tp1_acc[model_checkpoint]=' * FP16 PREC {top1} '.format(top1=acc_fp16)
            print(model_checkpoint,tp1_acc[model_checkpoint])

            # """quantize"""
            batch = evaluator.sample_batch()
            # input_list = [k for k in batch if k!="labels" ]
            # collate_fn  = lambda x: {k:x[k].cuda() for k in input_list}
            input_ids = batch.to(CFG_DEVICE)
            # ppq_quant_ir_FP8 = quantize_torch_model(
            #     model=model_fp16, calib_dataloader=evaluator.calib_dataloader, input_shape=input_ids.shape, input_dtype=input_ids.dtype,
            #     # model=model_fp16, calib_dataloader=evaluator.dataset, input_shape=input_ids.shape, input_dtype=input_ids.dtype,
            #     calib_steps=CALIB_STEP, collate_fn=evaluator.my_collate_fn, verbose=1,
            #     device=CFG_DEVICE, platform=CFG_PLATFORM_FP8, setting=QUANT_SETTING)

            """quantize int"""
            if os.path.exists(model_checkpoint[-6:]+'_layer_int8_wiki_v2.npy'):
                reports_int8 = np.load(model_checkpoint[-6:]+'_layer_int8_wiki_v2.npy',allow_pickle=True)
                reports_int8 = reports_int8.item()
            else:
                ppq_quant_ir_INT8 = quantize_torch_model(
                    model=model_fp16, calib_dataloader=evaluator.calib_dataloader, input_shape=input_ids.shape, input_dtype=input_ids.dtype,
                    # model=model_fp16, calib_dataloader=evaluator.dataset, input_shape=input_ids.shape, input_dtype=input_ids.dtype,
                    calib_steps=CALIB_STEP, collate_fn=evaluator.my_collate_fn, verbose=1,
                    device=CFG_DEVICE, platform=CFG_PLATFORM_INT8, setting=QUANT_SETTING)
                reports_int8 = layerwise_error_analyse_v2(
                    graph=ppq_quant_ir_INT8, running_device=CFG_DEVICE, collate_fn=evaluator.my_collate_fn, 
                    dataloader=evaluator.dataset)
                np.save(model_checkpoint[-6:]+'_layer_int8_wiki_v2',reports_int8)
                """evaluate"""
                executor = TorchExecutor(graph=ppq_quant_ir_INT8, device=CFG_DEVICE)
                model_forward_function = lambda input_tensor: torch.tensor(
                    executor(*[input_tensor])[0])
                acc_int8 = evaluator.evaluate_ppq(model_forward_function)
                tp1_acc[model_checkpoint]=' *  INT8 PREC {top5}'.format(top5=acc_int8)
                # tp1_acc[model_checkpoint]=' * INT8 PREC {top1} FP8 PREC {top3} MIX PREC {top5}'.format(top1=acc_int8, top3=acc_fp8 ,top5=acc_mix8)
                print(model_checkpoint,tp1_acc[model_checkpoint])

                executor = None
                ppq_quant_ir_INT8 = None

            """quantize fp"""
            if os.path.exists(model_checkpoint[-6:]+'_layer_fp8_wiki_v2.npy'):
                reports_fp8 = np.load(model_checkpoint[-6:]+'_layer_fp8_wiki_v2.npy',allow_pickle=True)
                reports_fp8 = reports_fp8.item()
            else:
                ppq_quant_ir_FP8 = quantize_torch_model(
                    model=model_fp16, calib_dataloader=evaluator.calib_dataloader, input_shape=input_ids.shape, input_dtype=input_ids.dtype,
                    # model=model_fp16, calib_dataloader=evaluator.dataset, input_shape=input_ids.shape, input_dtype=input_ids.dtype,
                    calib_steps=CALIB_STEP, collate_fn=evaluator.my_collate_fn, verbose=1,
                    device=CFG_DEVICE, platform=CFG_PLATFORM_FP8, setting=QUANT_SETTING)
                reports_fp8 = layerwise_error_analyse_v2(
                    graph=ppq_quant_ir_FP8, running_device=CFG_DEVICE, collate_fn=evaluator.my_collate_fn, 
                    dataloader=evaluator.dataset) 
                np.save(model_checkpoint[-6:]+'_layer_fp8_wiki_v2',reports_fp8)  
                """evaluate"""
                executor = TorchExecutor(graph=ppq_quant_ir_FP8, device=CFG_DEVICE)
                model_forward_function = lambda input_tensor: torch.tensor(
                    executor(*[input_tensor])[0])
                acc_fp8 = evaluator.evaluate_ppq(model_forward_function)
                tp1_acc[model_checkpoint]=' *  FP8 PREC {top5}'.format(top5=acc_fp8)
                # tp1_acc[model_checkpoint]=' * INT8 PREC {top1} FP8 PREC {top3} MIX PREC {top5}'.format(top1=acc_int8, top3=acc_fp8 ,top5=acc_mix8)
                print(model_checkpoint,tp1_acc[model_checkpoint])

                executor = None
                ppq_quant_ir_FP8 = None

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

            tp1_acc[model_checkpoint]=' FP16 PREC {top0} * INT8 PREC {top1} FP8 PREC {top3} MIX PREC {top5}'.format(
                top0=acc_fp16, top1=acc_int8, top3=acc_fp8 ,top5=acc_mix8)
            print(model_checkpoint,tp1_acc[model_checkpoint])
            
            executor = None
            ppq_quant_ir = None


        print(tp1_acc)
    else:
        raise Exception('You may not import this file.')
