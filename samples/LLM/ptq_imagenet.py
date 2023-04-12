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
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm

# PPQ_CONFIG.PPQ_DEBUG=True

CFG_DEVICE = 'cuda'                            # 一个神奇的字符串，用来确定执行设备
CFG_BATCHSIZE = 32                             # 测试与calib时的 batchsize
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
    # "distilbert-base-uncased-finetuned-sst-2-english",
    # "yoshitomo-matsubara/bert-base-uncased-sst2",
    # "yoshitomo-matsubara/bert-large-uncased-sst2",

    # "09panesara/distilbert-base-uncased-finetuned-cola",
    # "gchhablani/bert-base-cased-finetuned-cola",
    # "gchhablani/bert-large-cased-finetuned-cola",

    "gchhablani/bert-base-cased-finetuned-mnli",

    # "vicl/distilbert-base-uncased-finetuned-mrpc",
    # "gchhablani/bert-base-cased-finetuned-mrpc",
    # "gchhablani/bert-large-cased-finetuned-mrpc",

    # "anirudh21/distilbert-base-uncased-finetuned-rte",
    # "anirudh21/bert-base-uncased-finetuned-rte",

]
# seq = ["input_ids", "attention_mask", "token_type_ids", 
#         "position_ids", "head_mask", "inputs_embeds", 
#         "labels","output_attentions","output_hidden_states"]
tp1_acc={}

with ENABLE_CUDA_KERNEL():
    if __name__ == '__main__':

        """Load GLUE Dataset"""
        GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

        task = "mnli"
        actual_task = "mnli" if task == "mnli-mm" else task
        dataset = load_dataset("glue", actual_task)
        metric = load_metric('glue', actual_task)

        for model_checkpoint in model_list:
            # model_checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
            # model_checkpoint = "yoshitomo-matsubara/bert-base-uncased-sst2"
            # batch_size = 16
                
            """Preprocessing the data"""
            # tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
            task_to_keys = {
                "cola": ("sentence", None),
                "mnli": ("premise", "hypothesis"),
                "mnli-mm": ("premise", "hypothesis"),
                "mrpc": ("sentence1", "sentence2"),
                "qnli": ("question", "sentence"),
                "qqp": ("question1", "question2"),
                "rte": ("sentence1", "sentence2"),
                "sst2": ("sentence", None),
                "stsb": ("sentence1", "sentence2"),
                "wnli": ("sentence1", "sentence2"),
            }
            sentence1_key, sentence2_key = task_to_keys[task]
            def preprocess_function(examples):
                if sentence2_key is None:
                    return tokenizer(examples[sentence1_key], padding='max_length', truncation=True)
                return tokenizer(examples[sentence1_key], examples[sentence2_key], padding='max_length', truncation=True)
            tokenized_datasets = dataset.map(preprocess_function, batched=True)

            tokenized_datasets = tokenized_datasets.remove_columns([sentence1_key])
            if sentence2_key is not None:
                tokenized_datasets = tokenized_datasets.remove_columns([sentence2_key])
            tokenized_datasets = tokenized_datasets.remove_columns(["idx"])
            tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
            tokenized_datasets.set_format("torch")
            print("dataset len: ",len(tokenized_datasets["train"]),len(tokenized_datasets["validation_matched"]))
            small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1600))
            small_eval_dataset = tokenized_datasets["validation_matched"].shuffle(seed=42).select(range(8000))
            # small_train_dataset = tokenized_datasets["train"]
            # small_eval_dataset = tokenized_datasets["validation"]
            train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=CFG_BATCHSIZE)
            # train_dataloader = DataLoader(small_train_dataset, batch_size=CFG_BATCHSIZE)
            eval_dataloader = DataLoader(small_eval_dataset, batch_size=CFG_BATCHSIZE)

            """Eval the original model"""
            num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
            model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
            # metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
            model_name = model_checkpoint.split("/")[-1]
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            model.to(device)        
            model.eval()
            for batch in eval_dataloader:
                # print(batch)
                batch = {k: v.to(device) for k, v in batch.items()}
                # print(batch["token_type_ids"].size())
                # print(batch["token_type_ids"],batch["token_type_ids"].dtype,batch["token_type_ids"].shape)
                # break
                with torch.no_grad():
                    outputs = model(**batch)
                # print("OUTPUT: ",outputs)
                # break
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predictions.tolist(), references=batch["labels"].tolist())
            full_pre_res = metric.compute()
            print(full_pre_res)
            
            """quantize"""
            # dataloader = load_imagenet_from_directory(
            #     directory=CFG_TRAIN_DIR, batchsize=CFG_BATCHSIZE,
            #     shuffle=False, subset=5120, require_label=False,
            #     num_of_workers=8)
            for batch in train_dataloader:
                break
            # print(batch)
            input_list = [k for k in batch if k!="labels" ]
            collate_fn  = lambda x: {k:x[k].cuda() for k in input_list}
            inputs = { k:batch[k].cuda() for k in input_list}
            # collate_fn  = lambda x: {'input_ids': x['input_ids'].cuda(), 'attention_mask': x['attention_mask'].cuda()}
            # inputs = {'input_ids': batch['input_ids'].cuda(), 'attention_mask': batch['attention_mask'].cuda()}
            print(input_list)
            ppq_quant_ir = quantize_torch_model(
                model=model, calib_dataloader=train_dataloader, input_shape=None, inputs=inputs, input_dtype=None,
                calib_steps=len(small_train_dataset) // CFG_BATCHSIZE, collate_fn=collate_fn, verbose=1,
                device=CFG_DEVICE, platform=CFG_PLATFORM, setting=QUANT_SETTING)
            # for op_name, operation in ppq_quant_ir.operations.items():
            #     print(op_name,operation.platform)
            # print("embeddings: ",model.embeddings)

            """evaluate"""
            executor = TorchExecutor(graph=ppq_quant_ir, device=CFG_DEVICE)
            model_forward_function = lambda input_tensor: torch.tensor(
                executor(*[input_tensor])[0])
            # model_forward_function = lambda input_tensor: executor(*[input_tensor])
            metric = load_metric('glue', actual_task)
            for batch_idx, batch in tqdm(enumerate(eval_dataloader), 
                    desc='Evaluating Model...', total=len(eval_dataloader)):
                batch = {k: v.to(device) for k, v in batch.items()}
                # print(batch["token_type_ids"].size())
                # print(batch["token_type_ids"],batch["token_type_ids"].dtype,batch["token_type_ids"].shape)
                # break

                b_inputs = {k:v for k, v in batch.items() if k!="labels"}
                # print(b_inputs["input_ids"].shape,b_inputs["attention_mask"].shape)
                batch_pred = model_forward_function(b_inputs)
                # print(batch_pred)
                # break

                # print(batch_pred.shape)
                predictions = torch.argmax(batch_pred, dim=-1)
                metric.add_batch(predictions=predictions.tolist(), references=batch["labels"].tolist())
            
            quant_res = metric.compute()
            print(quant_res)
            tp1_acc[model_checkpoint]=' * FULL PREC {top1} QUANT PREC {top5} PLATFORM '.format(
                top1=full_pre_res, top5=quant_res)
            print(tp1_acc[model_checkpoint])

        print(tp1_acc)
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
        # report.to_csv('efficientnet_v2l_int8.csv')

        # #从大到小排序单层误差
        # sensitivity = [(op_name, error) for op_name, error in reports.items()]
        # sensitivity = sorted(sensitivity, key=lambda x: x[1], reverse=True)

        # # 将前十个误差最大的层送上 FP32
        # for op_name, _ in sensitivity[: 20]:
        #     QUANT_SETTING.dispatching_table.append(operation=op_name, platform=TargetPlatform.FP32)
        # # QUANT_SETTING.dispatching_table.append(operation='Conv_0',platform=TargetPlatform.FP32)
        # ppq_quant_ir = quantize_torch_model(
        #     model=model, calib_dataloader=dataloader, input_shape=CFG_INPUT_SHAPE,
        #     calib_steps=5120 // CFG_BATCHSIZE, collate_fn=lambda x: x.to(CFG_DEVICE), verbose=1,
        #     device=CFG_DEVICE, platform=CFG_PLATFORM, setting=QUANT_SETTING)

        # ppq_int8_report = evaluate_ppq_module_with_imagenet(
        #     model=ppq_quant_ir, imagenet_validation_dir=CFG_VALIDATION_DIR,
        #     batchsize=CFG_BATCHSIZE, device=CFG_DEVICE, verbose=True)

        """output"""
        # print(type(ppq_int8_report),ppq_int8_report)
        # tp1_acc[model_name]=' * Prec@1 {top1:.3f} Prec@5 {top5:.3f}'.format(
        # top1=sum(ppq_int8_report['top1_accuracy'])/len(ppq_int8_report['top1_accuracy']), 
        # top5=sum(ppq_int8_report['top5_accuracy'])/len(ppq_int8_report['top5_accuracy']))
        # print(tp1_acc[model_name])

        # export_ppq_graph(
        #     graph=ppq_quant_ir, 
        #     platform=TargetPlatform.ONNXRUNTIME,
        #     graph_save_to=f'{os.path.join(CFG_DUMP_PATH, model_name)}.onnx')
        
        # evaluate_onnx_module_with_imagenet(
        #     onnxruntime_model_path=f'{os.path.join(CFG_DUMP_PATH, model_name)}.onnx', 
        #     imagenet_validation_dir=CFG_VALIDATION_DIR, batchsize=CFG_BATCHSIZE, 
        #     device=CFG_DEVICE)
        # print(tp1_acc)
    else:
        raise Exception('You may not import this file.')
