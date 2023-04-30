
from typing import Callable, Dict, Iterator, List

import torch
from ppq.core import PASSIVE_OPERATIONS, ppq_warning
from ppq.executor import RuntimeHook, TorchExecutor
from ppq.IR import BaseGraph, Operation, QuantableOperation, Variable
from ppq.quantization.measure.norm import torch_snr_error
from ppq.utils.fetch import batch_random_fetch, tensor_random_fetch
from tqdm import tqdm

from .util import MeasurePrinter, MeasureRecorder


class OutputRecorder_v2(RuntimeHook):
    def __init__(self, operation: Operation, fetchs: int = 4096) -> None:
        self.fetched     = None
        self.fetchs      = fetchs
        self.fp_inputs = []
        self.device_list = []
        self.analyse = False
        self.batch_id = 0
        super().__init__(operation)

    def pre_forward_hook(self, inputs: list, **kwargs) -> list:
        if not self.analyse:
            # self.fp_inputs.append(inputs)
            cache_inputs = [i.to('cpu') for i in inputs]
            device_list = [i.device for i in inputs]
            self.fp_inputs.append(cache_inputs)
            self.device_list.append(device_list)
            # print("memory_allocated",torch.cuda.memory_allocated())
            # print("memory_catched",torch.cuda.memory_reserved())
        else:
            inputs = [i.to(device) for i,device in zip(self.fp_inputs[self.batch_id],
                    self.device_list[self.batch_id])]
            self.batch_id += 1
        return super().pre_forward_hook(inputs, **kwargs)

    def post_forward_hook(self, outputs: list, **kwargs) -> list:
        output_tensor = outputs[0]
        assert isinstance(output_tensor, torch.Tensor), (
            'Output of monitoring operation is not a torch.Tensor')
        self.fetched = batch_random_fetch(
            output_tensor, seed=10086, fetchs_per_batch=self.fetchs
        ).to('cpu')
        return super().post_forward_hook(outputs, **kwargs)

    def pop(self) -> torch.Tensor:
        fetched = self.fetched
        self.fetched = None
        return fetched
    
    def convert(self):
        self.analyse = True


def layerwise_error_analyse_v2(
    graph: BaseGraph, running_device: str,
    dataloader: Iterator, collate_fn: Callable = None, method: str = 'snr',
    steps: int = 20, verbose: bool = True, fetchs: int = 4096) -> Dict[str, float]:
    """Measure the difference from a quantized graph to its dequantized graph.

    A dictionary contains output differences for all operation will be returned as a result.

        Result is like: {'operation name 1': 0.933, 'operation name 2': 0.926}

    if verbose is set as True, this function will display error report at last.

    The key of the dictionary is an operation name while the value of corresponding key
        is the difference between quantized output and float output of this operation.

    Result {'operation name 1': 0.933} means that quantized graph and fp32 graph have a difference
        (or similarity, based on your measurement) 0.933 at the output tensor of 'operation name 1'.

    ATTENTION: Output difference is measured at graph-level, it includes the difference accmulated from the
        very beginning operation along to the target operation.

    Args:
        graph (BaseGraph):
            A fully quantized graph instance.

        running_device (str):
            A device string used to initialize a graph executor for the graph execution.
                if a executor was given, this parameter will be skipped.

        dataloader (Iterator):
            Test dataloader, this function will measure the output difference based on given data.

        collate_fn (Callable, optional):
            An data preprocessing function provided by user to convert data from dataloader towards
                executable format. If set as None, then no action will be taken during preprocessing.

        method (str, optional):
            A string indicates a measurement to calculate the difference of quantized output and fp32 one.
                'cosine', 'snr', and 'mse' is supported in PPQ for now.

        steps (Int, optional)
            computation steps.

    Returns:
        A dictionary contains output differences for all operation will be returned from this function.

        Result is like: {'operation name 1': 0.933, 'operation name 2': 0.926}
    """
    executor = TorchExecutor(graph=graph, device=running_device)

    # find all quantable operations.
    interested_op = [operation for operation in graph.operations.values()
                     if (isinstance(operation, QuantableOperation) and operation.is_computing_op)]
    if len(interested_op) == 0:
        print('Oops. you got nothing to analyse.')
        return

    # set up all hooks.
    recorders, hooks, caches = {}, {}, {}
    for operation in interested_op:
        if isinstance(operation, QuantableOperation):
            if operation.num_of_output > 1:
                ppq_warning(f'Operation {operation.name} has more than 1 output, '
                            'analyser will process the first output of it.')

            recorders[operation.name] = MeasureRecorder(measurement=method)
            hooks[operation.name] = OutputRecorder_v2(
                operation=operation, fetchs=fetchs)
            caches[operation.name] = []

    # dequantize all
    for operation in graph.operations.values():
        if isinstance(operation, QuantableOperation):
            operation.dequantize()

    # run for each quantable operations:
    for idx, batch in tqdm(enumerate(dataloader),
                           desc='Analysing Graphwise Quantization Error(Phrase 1):',
                           total=(min(len(dataloader), steps))):
        if collate_fn is not None: batch = collate_fn(batch)
        executor.forward(inputs=batch, hooks=hooks)

        for operation in interested_op:
            hook = hooks[operation.name]
            caches[operation.name].append(hook.pop())

        if idx >= steps: break

    # restore all
    for operation in graph.operations.values():
        if isinstance(operation, QuantableOperation):
            operation.restore_quantize_state()
    for op_name in hooks:
        hooks[op_name].analyse=True

    # run for each quantable operations:
    for idx, batch in tqdm(enumerate(dataloader),
                           desc='Analysing Graphwise Quantization Error(Phrase 2):',
                           total=(min(len(dataloader), steps))):
        if collate_fn is not None: batch = collate_fn(batch)
        executor.forward(inputs=batch, hooks=hooks)

        for operation in interested_op:
            recorder = recorders[operation.name]
            hook     = hooks[operation.name]
            cache    = caches[operation.name]
            recorder.update(y_real = cache[idx], y_pred = hook.pop())

        if idx >= steps: break

    results = {}
    for operation in interested_op:
        assert isinstance(operation, QuantableOperation)
        results[operation.name] = recorders[operation.name].measure

    if verbose:
        method_str = 'MEASUREMENT'
        if method == 'snr': method_str = 'NOISE:SIGNAL POWER RATIO'
        if method == 'cosine': method_str = 'COSINE SIMILARITY'
        if method == 'mse': method_str = 'MSE LOSS(UNSCALED)'
        MeasurePrinter(results, order='large_to_small', measure=method_str, percentage=method in {'snr', 'cosine'}).print()
    return results
