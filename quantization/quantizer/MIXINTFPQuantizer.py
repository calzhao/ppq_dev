from typing import Union

import torch
from ppq.core import (PASSIVE_OPERATIONS, OperationQuantizationConfig,
                      QuantizationPolicy, QuantizationProperty,
                      QuantizationStates, RoundingPolicy, TargetPlatform)
from ppq.IR import BaseGraph, Operation

from .base import BaseQuantizer


class MIXINTFPQuantizer(BaseQuantizer):
    def __init__(
        self, graph: BaseGraph
    ) -> Union[torch.Tensor, list, dict]:
        super().__init__(graph=graph)
        self._num_of_bits = 8
        self._quant_min = - int(pow(2, self._num_of_bits - 1))
        self._quant_max = int(pow(2, self._num_of_bits - 1) - 1)

    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        base_quant_config = self.create_default_quant_config(
            policy=self.quantize_policy, rounding=self.rounding_policy,
            op=operation, num_of_bits=self._num_of_bits, exponent_bits=0,
            quant_max=self._quant_max, quant_min=self._quant_min,
            observer_algorithm='percentile')

        if operation.platform==TargetPlatform.TRT_FP8:
            base_quant_config = self.create_fp_quant_config(operation=operation)

        if operation.type in {'Conv', 'ConvTranspose', 'Gemm', 'MatMul'}:
            # set all parameters within Conv, ConvTranspose, Gemm to per-channel quant-config.
            assert operation.num_of_input > 0, 'Seems you got a Conv layer with no parameters.'

            # first parameter must exits, for conv layer it will be conv_weight
            # layout: [out_channel, in_channel, kernel_size, kernel_size]
            if operation.type in {'Conv', 'ConvTranspose'}:
                conv_weight_config = base_quant_config.input_quantization_config[1]
                if operation.platform==TargetPlatform.TRT_FP8:
                    conv_weight_config.policy = QuantizationPolicy(
                        QuantizationProperty.SYMMETRICAL +
                        QuantizationProperty.PER_CHANNEL + 
                        QuantizationProperty.FLOATING +
                        QuantizationProperty.POWER_OF_2
                    )
                    conv_weight_config.observer_algorithm = 'floating'
                else:
                    conv_weight_config.policy = QuantizationPolicy(
                        QuantizationProperty.SYMMETRICAL +
                        QuantizationProperty.LINEAR +
                        QuantizationProperty.PER_CHANNEL
                    )
                    conv_weight_config.observer_algorithm = 'minmax'
                conv_weight_config.channel_axis = (1 if operation.type == 'ConvTranspose' else 0)

            # first parameter must exits, for gemm layer it will be gemm_weight
            # layout: [in_dim, out_dim]
            elif operation.type in {'Gemm', 'MatMul'}:
                gemm_weight_config = base_quant_config.input_quantization_config[1]
                if operation.platform==TargetPlatform.TRT_FP8:
                    gemm_weight_config.policy = QuantizationPolicy(
                        QuantizationProperty.SYMMETRICAL +
                        QuantizationProperty.PER_CHANNEL +
                        QuantizationProperty.FLOATING +
                        QuantizationProperty.POWER_OF_2
                    )
                    gemm_weight_config.observer_algorithm = 'floating'
                else:
                    gemm_weight_config.policy = QuantizationPolicy(
                        QuantizationProperty.SYMMETRICAL +
                        QuantizationProperty.LINEAR +
                        QuantizationProperty.PER_CHANNEL
                    )
                    gemm_weight_config.observer_algorithm = 'minmax'
                gemm_weight_config.channel_axis = 0
            # if operation has bias
            if operation.num_of_input > 2:
                if operation.platform==TargetPlatform.TRT_FP8:
                    bias_config = base_quant_config.input_quantization_config[-1]
                    bias_config.state = QuantizationStates.FP32
                else:
                    bias_config = base_quant_config.input_quantization_config[-1]
                    bias_config.policy = QuantizationPolicy(
                        QuantizationProperty.SYMMETRICAL +
                        QuantizationProperty.LINEAR +
                        QuantizationProperty.PER_CHANNEL
                    )
                    bias_config.num_of_bits = 32
                    bias_config.quant_max = int(pow(2, bias_config.num_of_bits - 1)) - 1
                    bias_config.quant_min = - int(pow(2, bias_config.num_of_bits - 1)) + 1
                    bias_config.state = QuantizationStates.PASSIVE_INIT
                    bias_config.channel_axis = 0
                    bias_config.observer_algorithm = 'minmax'
            
            # if operation.platform==TargetPlatform.TRT_FP8:
            #     for output_config in base_quant_config.output_quantization_config:
            #         output_config.state = QuantizationStates.FP32

            # 所有算子只量化输入
            for output_config in base_quant_config.output_quantization_config:
                output_config.state = QuantizationStates.FP32

        if operation.type in PASSIVE_OPERATIONS:
            # Those op are not active op.
            base_quant_config.is_active_quant_op = False
        return base_quant_config

    def create_fp_quant_config(self,operation: Operation)  -> OperationQuantizationConfig:
        return self.create_default_quant_config(
            policy=QuantizationPolicy(
            QuantizationProperty.SYMMETRICAL +
            QuantizationProperty.PER_TENSOR +
            QuantizationProperty.FLOATING +
            QuantizationProperty.POWER_OF_2),
            rounding=self.rounding_policy, op=operation, num_of_bits=self._num_of_bits, 
            exponent_bits=4, quant_max= 448.0, quant_min= - 448.0, observer_algorithm='floating')

    @ property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.PPL_CUDA_INT8

    @ property
    def default_platform(self) -> TargetPlatform:
        return TargetPlatform.FP32

    @ property
    def quant_operation_types(self) -> set:
        return {
            # 'Conv', 'Relu', 'PRelu', 'Clip', 'Gemm',
            # 'Resize', 'MaxPool', 'AveragePool',
            # 'GlobalMaxPool', 'GlobalAveragePool',
            # 'Mul', 'Add', 'LeakyRelu', 'Split', 'Concat',
            # 'Transpose', 'Slice', 'Reshape', 'Flatten',
            # 'Sigmoid', 'ReduceMean'
            
            # Align with FP8
            'Conv', 'Gemm',
            'AveragePool',
            'GlobalAveragePool',
            'ConvTranspose',
            'MatMul'
        }

    @ property
    def quantize_policy(self) -> QuantizationPolicy: # 需要改进
        return QuantizationPolicy(
            QuantizationProperty.SYMMETRICAL +
            QuantizationProperty.LINEAR +
            QuantizationProperty.PER_TENSOR
        )

    @ property
    def rounding_policy(self) -> RoundingPolicy:
        return RoundingPolicy.ROUND_HALF_EVEN

    @ property
    def activation_fusion_types(self) -> set:
        return {'Relu', 'Clip', 'Sigmoid', 'LeakyRelu'}
