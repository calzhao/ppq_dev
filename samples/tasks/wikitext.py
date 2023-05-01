
"""
    wikitext 实现
"""
import torch
import torch.nn.functional as F
from lm_eval import utils
from lm_eval.metrics import mean, weighted_perplexity, weighted_mean, bits_per_byte
import random


class WikiEvaluator:
    def __init__(self, dataset, tokenizer, device, _model_call, CALIB_STEP):
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