import torch
from ppq.core import (PPQ_CONFIG, QuantizationProperty, QuantizationStates,
                      RoundingPolicy, TensorQuantizationConfig)
from torch.autograd import Function
import struct
import numpy as np

def uint2fp(s):
    return struct.unpack('f',struct.pack('I',s))[0]

def fp2uint(s):
    return struct.unpack('I',struct.pack('f',s))[0]

class TensorwiseFloatingQuantImpl(Function):
    """Torch Tensorwise quantize is designed to quantize a torch Tensor
    with a given configuration. All quantization within PPQ will invoke
    this function to quantize its value. Any modification of this function
    will greatly affects system behaviour.

    This is a torch implementation of quantization itself.
    Notice that if ppq.config.USING_CUDA_KERNAL = True,
        then all quantization will use ffi.CUDA instead.

    Notice this function will always clone your tensor value first.
    This function never quantize your tensor value inplace.
    """
    @ staticmethod
    def forward(ctx, tensor: torch.Tensor, scales: torch.Tensor, offsets: torch.Tensor,
                exponet_bits: int, mantissa_bits: int,
                quant_min: float, quant_max: float,
                rounding: RoundingPolicy) -> torch.Tensor:
        scales, offsets = scales.to(tensor.device), offsets.to(tensor.device)
        if not PPQ_CONFIG.USING_CUDA_KERNEL or not tensor.is_cuda:
            # quantization function, pytorch implmentation
            # raise NotImplementedError('This Feature must run with PPQ Cuda Kernel.')
            Unscaled_FP32 = tensor / scales

            # helper_value = Unscaled_FP32;
            exponent_min  = -(1 << (exponet_bits - 1)) + 1
            exponent_max  = (1 << (exponet_bits - 1))

            fp32_sign = 0
            fp32_exp  = (exponent_max + 127) << 23
            fp32_mantissa = ~(0x007FFFFF >> mantissa_bits) & 0x007FFFFF;
            helper_data = fp32_sign + fp32_mantissa + fp32_exp
            theoretical_maximum = uint2fp(helper_data)

            tensor = torch.clamp(Unscaled_FP32,max(quant_min, -theoretical_maximum), 
                    min(quant_max, theoretical_maximum))

            def elew(Unscaled_FP32):
                helper_data   = fp2uint(Unscaled_FP32);
                fp32_sign     = helper_data & 0x80000000;
                fp32_exp      = helper_data & 0x7F800000;
                fp32_mantissa = helper_data & 0x007FFFFF;
                
                if (((fp32_exp >> 23) - 127) < exponent_min + 1):
                    min_subnormal = 1.0 / (1 << ((1 << (exponet_bits - 1)) + mantissa_bits - 2))
                    tensor =  round(Unscaled_FP32 / min_subnormal) * min_subnormal
                else:
                    rounding_helper_data = ((fp32_mantissa << (mantissa_bits)) & 0x007FFFFF) + 0x3F800000
                    round_bit = round(uint2fp(rounding_helper_data) - 1)

                    fp32_mantissa = ((fp32_mantissa >> (23 - mantissa_bits)) + round_bit) << (23 - mantissa_bits)
                    helper_data = fp32_sign + fp32_mantissa + fp32_exp
                    tensor = np.clip(uint2fp(helper_data), quant_min, quant_max)
                return tensor
            

            return tensor.apply_(elew)
        
        else:
            from ppq.core import CUDA

            # quantization function, pure cuda implmentation
            quantized = CUDA.FloatingQuantize_T(
                tensor=tensor,
                scales=scales,
                offsets=offsets,
                exponent=exponet_bits,
                mantissa=mantissa_bits,
                minimum=quant_min,
                maximum=quant_max,
                rounding=rounding.value
            )
            return quantized

    @ staticmethod
    def backward(ctx, dy: torch.Tensor):
        return dy, None, None, None, None, None, None, None, None


class ChannelwiseFloatingQuantImpl(Function):
    """Torch Channelwise quantize is designed to quantize a torch Tensor
    with a given configuration. All quantization within PPQ will invoke
    this function to quantize its value. Any modification of this function
    will greatly affects system behaviour.

    This is a torch implementation of quantization itself.
    Notice that if ppq.config.USING_CUDA_KERNAL = True,
        then all quantization will bypass this function by using ffi.CUDA instead.

    Notice this function will always clone your tensor value first.
    This function never quantize your tensor value inplace.
    """
    @ staticmethod
    def forward(ctx, tensor: torch.Tensor, scales: torch.Tensor,
                offsets: torch.Tensor, channel_axis: int,
                exponet_bits: int, mantissa_bits: int, 
                quant_min: float, quant_max: float,
                rounding: RoundingPolicy) -> torch.Tensor:

        scales, offsets = scales.to(tensor.device), offsets.to(tensor.device)
        if not PPQ_CONFIG.USING_CUDA_KERNEL or not tensor.is_cuda:
            # generate a shape that likes [1, 1, -1, 1], the only -1 is at channel axe.
            # raise NotImplementedError('This Feature must run with PPQ Cuda Kernel.')
            shape = [1 if axis != channel_axis else -1 for axis in range(tensor.ndim)]
            scale = scales.view(shape)

            Unscaled_FP32 = tensor / scale

            # helper_value = Unscaled_FP32;
            exponent_min  = -(1 << (exponet_bits - 1)) + 1
            exponent_max  = (1 << (exponet_bits - 1))

            fp32_sign = 0
            fp32_exp  = (exponent_max + 127) << 23
            fp32_mantissa = ~(0x007FFFFF >> mantissa_bits) & 0x007FFFFF;
            helper_data = fp32_sign + fp32_mantissa + fp32_exp
            theoretical_maximum = uint2fp(helper_data)

            tensor = torch.clamp(Unscaled_FP32,max(quant_min, -theoretical_maximum), 
                min(quant_max, theoretical_maximum))

            def elew(Unscaled_FP32):
                helper_data   = fp2uint(Unscaled_FP32);
                fp32_sign     = helper_data & 0x80000000;
                fp32_exp      = helper_data & 0x7F800000;
                fp32_mantissa = helper_data & 0x007FFFFF;
                
                if (((fp32_exp >> 23) - 127) < exponent_min + 1):
                    min_subnormal = 1.0 / (1 << ((1 << (exponet_bits - 1)) + mantissa_bits - 2))
                    tensor =  round(Unscaled_FP32 / min_subnormal) * min_subnormal
                else:
                    rounding_helper_data = ((fp32_mantissa << (mantissa_bits)) & 0x007FFFFF) + 0x3F800000
                    round_bit = round(uint2fp(rounding_helper_data) - 1)

                    fp32_mantissa = ((fp32_mantissa >> (23 - mantissa_bits)) + round_bit) << (23 - mantissa_bits)
                    helper_data = fp32_sign + fp32_mantissa + fp32_exp
                    tensor = np.clip(uint2fp(helper_data), quant_min, quant_max)
                return tensor
            

            return tensor.apply_(elew)
        else:
            from ppq.core import CUDA
            quantized = CUDA.FloatingQuantize_C(
                tensor=tensor,
                scales=scales,
                offsets=offsets,
                channel_axis=channel_axis,
                exponent=exponet_bits,
                mantissa=mantissa_bits,
                minimum=quant_min,
                maximum=quant_max,
                rounding=rounding.value)
            return quantized

    @ staticmethod
    def backward(ctx, dy: torch.Tensor):
        return dy, None, None, None, None, None, None, None, None, None


def PPQFloatingQuantFunction(
    tensor: torch.Tensor, config: TensorQuantizationConfig) -> torch.Tensor:
    if not PPQ_CONFIG.USING_CUDA_KERNEL:
        raise PermissionError('PPQ Floating Quant Function require PPQ_CONFIG.USING_CUDA_KERNEL = True')
    # if not tensor.is_cuda:
    #     raise PermissionError('PPQ Floating Quant Function requires tensor device to be cuda, '
    #                           'CPU floating quantization is not implemented yet.')

    """PPQ 核心量化函数，没啥好说的了吧，这个玩意既做 quant 也做 dequant"""
    if not QuantizationStates.is_activated(config.state): return tensor
    if not config.policy.has_property(QuantizationProperty.FLOATING):
        raise ValueError('Critical Quantization Error! Unexpected policy detected. '
                         'PPQFloatingQuantFunction except a Floating Quantization Config.')
    if config.policy.has_property(QuantizationProperty.DYNAMIC):
        raise ValueError('Unexpected Dynamic Flag in Quantization Policy.')

    if config.policy.has_property(QuantizationProperty.PER_CHANNEL):
        return ChannelwiseFloatingQuantImpl.apply(
            tensor, config.scale, config.offset, config.channel_axis,
            config.exponent_bits, config.mantissa_bits,
            config.quant_min, config.quant_max, config.rounding)
    elif config.policy.has_property(QuantizationProperty.PER_TENSOR):
        return TensorwiseFloatingQuantImpl.apply(
            tensor, config.scale, config.offset,
            config.exponent_bits, config.mantissa_bits,
            config.quant_min, config.quant_max, config.rounding)
