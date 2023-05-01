"""
    lambada 实现
"""
import torch

tp1_acc={}
class LambadaEvaluator:
    def __init__(self, dataset, tokenizer, device, model_call, CALIB_STEP):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.model_call = model_call

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'],padding='longest',truncation=True)
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        # print(self.dataset[0])
        self.dataset.set_format(type='torch', columns=['input_ids','attention_mask'])
        self.calib_dataloader=self.dataset.shuffle(seed=29).select(range(CALIB_STEP))

    @torch.no_grad()
    def sample_batch(self):
        return self.dataset[0]['input_ids'].to(self.device).unsqueeze(0)

    @torch.no_grad()
    def my_collate_fn(self, batch):
        return batch['input_ids'].to(self.device).unsqueeze(0)

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
    
    @torch.no_grad()
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
