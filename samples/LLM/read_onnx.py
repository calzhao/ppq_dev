import onnx
from ppq.api import load_graph
from ppq.core import (PPQ_CONFIG, NetworkFramework, TargetPlatform,
                      empty_ppq_cache, ppq_warning)
import torch.onnx
import torch.nn as nn          
# Load the ONNX model
ppq_cpu = onnx.load("cpuexport.model")
ppq_gpu = onnx.load("gpuexport.model")
assert isinstance(ppq_cpu, onnx.ModelProto)
assert isinstance(ppq_gpu, onnx.ModelProto)
# pytorch_model = nn.
# Print the model metadata
# print("Model metadata:")
# print("  Graph name:", model.graph.name)
# print("  Graph inputs:", [input.name for input in model.graph.input])
# print("  Graph outputs:", [output.name for output in model.graph.output])
# print("  Graph nodes:")

for node, gpu_node in zip(ppq_cpu.graph.node,ppq_gpu.graph.node):
    # print(node.name)
    print("  cpu  ", node.op_type, node.name, [input for input in node.input], [output for output in node.output])
    print("  gpu  ", gpu_node.op_type, gpu_node.name, [input for input in gpu_node.input], [output for output in gpu_node.output])
# for node in graph_pb.node:
#     op_name = node.name
# print("ppq_gpu",ppq_gpu.graph)
# print("ppq_cpu",ppq_cpu.graph)
# print("ppq_gpu",ppq_gpu)