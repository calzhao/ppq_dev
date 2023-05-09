"""
    piqa 实现
"""
import torch
import torch.nn.functional as F

class PiqaEvaluator:
    def __init__(self, dataset, tokenizer, device, _model_call, CALIB_STEP):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self._model_call=_model_call
        self.padding_length = 0

        def set_padding(examples):
            goal_len = max(len(elem) for elem in examples['goal'])
            choice1_len = max(len(elem) for elem in examples['sol1'])
            choice2_len = max(len(elem) for elem in examples['sol2'])
            self.padding_length = max(max(goal_len+choice1_len,goal_len+choice2_len)+20,self.padding_length)
            return None
        self.dataset.map(set_padding, batched=True)
        
        def tokenize_function(examples):
            out_doc = self._process_doc(examples)
            out_doc['context_enc'] = self.tokenizer(self.doc_to_text(out_doc),truncation=True)
            out_doc['continuation_enc1'] = self.tokenizer(self.doc_to_target1(out_doc),truncation=True)
            out_doc['continuation_enc2'] = self.tokenizer(self.doc_to_target2(out_doc),truncation=True)
            return out_doc
        
        self.dataset = self.dataset.map(tokenize_function, batched=False)
        self.calib_dataloader=self.dataset.shuffle(seed=29).select(range(CALIB_STEP))
        print(self.padding_length, self.dataset[0])

    def doc_to_text(self, doc):
        return "Question: " + doc["goal"] + "\nAnswer:"
    def doc_to_target1(self, doc):
        return " " + doc["choices"][0]
    def doc_to_target2(self, doc):
        return " " + doc["choices"][1]
    def _process_doc(self, doc):
        out_doc = {
            "goal": doc["goal"],
            "choices": [doc["sol1"], doc["sol2"]],
            "gold": doc["label"],
        }
        return out_doc
    def _loglikelihood_tokens(self, context_enc, continuation_enc, foward_func=None, sample=False):

        padding_length = self.padding_length

        # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
        # tensors, then we pack them together into a batch, call the model, and then pick it all apart
        # again because vectorizing is annoying

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

        # Slice to original seq length
        contlen = len(cont_toks)
        logits = logits[:,inplen - contlen : inplen]  # [1, seq, vocab]

        # Check if per-token argmax is exactly equal to continuation
        greedy_tokens = logits.argmax(dim=-1)
        cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(
            0
        )  # [1, seq]
        max_equal = (greedy_tokens == cont_toks).all()

        # Obtain log-probs at the corresponding continuation token indices
        logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
            -1
        )  # [1, seq]

        # Answer: (log prob, is-exact-match)
        answer = (float(logits.sum()), bool(max_equal))

        return answer
    
    @torch.no_grad()
    def sample_batch(self):
        return torch.ones((1,self.padding_length),dtype=torch.int32)

    @torch.no_grad()
    def my_collate_fn(self, batch):
        context_enc = batch['context_enc']['input_ids']
        continuation_enc1 = batch['continuation_enc1']['input_ids']
        input1 = self._loglikelihood_tokens(context_enc,continuation_enc1,sample=True)
        return input1.to(self.device)

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in self.dataset:
            context_enc = batch['context_enc']['input_ids']
            continuation_enc1 = batch['continuation_enc1']['input_ids']
            continuation_enc2 = batch['continuation_enc2']['input_ids']
            label = batch['gold']
            outputs1 = self._loglikelihood_tokens(context_enc,continuation_enc1)[0]
            outputs2 = self._loglikelihood_tokens(context_enc,continuation_enc2)[0]
            pred = 0 if outputs1 > outputs2 else 1
            total += 1
            hit += pred == label
        acc = hit / total
        return acc
    
    def evaluate_ppq(self, fw_func):
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in self.dataset:
            context_enc = batch['context_enc']['input_ids']
            continuation_enc1 = batch['continuation_enc1']['input_ids']
            continuation_enc2 = batch['continuation_enc2']['input_ids']
            label = batch['gold']
            outputs1 = self._loglikelihood_tokens(context_enc,continuation_enc1,foward_func=fw_func)[0]
            outputs2 = self._loglikelihood_tokens(context_enc,continuation_enc2,foward_func=fw_func)[0]
            pred = 0 if outputs1 > outputs2 else 1
            total += 1
            hit += pred == label
        acc = hit / total
        return acc