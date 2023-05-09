"""
    hellaswag 实现
"""
import torch
import torch.nn.functional as F
import re
import numpy as np

class HellaEvaluator:
    def __init__(self, dataset, tokenizer, device, _model_call, CALIB_STEP):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self._model_call =_model_call
        self.padding_length = 0
        
        def tokenize_function(examples):
            out_doc = self._process_doc(examples)
            out_doc['context_enc'] = self.tokenizer(self.doc_to_text(out_doc),truncation=True)
            out_doc['continuation_enc'] = self.tokenizer(self.doc_to_target(out_doc),truncation=True)
            return out_doc
        
        self.dataset = self.dataset.map(tokenize_function, batched=False)
        self.calib_dataloader=self.dataset.shuffle(seed=29).select(range(CALIB_STEP))
        
        def set_padding(examples):
            # print(examples)
            context_enc = max(len(elem['input_ids']) for elem in examples['context_enc'])
            # print([max(map(len, ele['input_ids'])) for ele in examples['continuation_enc']])
            continuation_enc = max([max(map(len, ele['input_ids'])) for ele in examples['continuation_enc'] ])
            self.padding_length = max(context_enc+continuation_enc+20, self.padding_length)
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