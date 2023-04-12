from typing import Iterable
import torch
import torchvision
from torch.utils.data import DataLoader

from ppq import BaseGraph, QuantizationSettingFactory, TargetPlatform
from ppq.api import export_ppq_graph, quantize_torch_model

DEVICE = 'cuda'
PLATFORM = TargetPlatform.PPL_CUDA_INT8 # identify a target platform for your network.

batch = 1
channels = 3
size = 32

# diffusion model
from diffusers import DiffusionPipeline
from diffusers.utils import floats_tensor

generator = DiffusionPipeline.from_pretrained("google/ddpm-cat-256")
#generator.to("cuda")
model = generator.unet
model = model.to(DEVICE)

#def load_calibration_dataset() -> Iterable:
#    return [torch.rand(size=INPUT_SHAPE) for _ in range(32)]

quant_setting = QuantizationSettingFactory.default_setting()

# Load training data for creating a calibration dataloader.

def dummy_input():
    noise = floats_tensor((batch,channels,size,size))
    time_step = 1
    time_step = torch.tensor([1])
    return {"sample": noise, "timestep": time_step}

def load_calibration_dataset() -> Iterable:
    return [dummy_input()]
calibration_dataset = load_calibration_dataset()

calibration_dataloader = DataLoader(
    dataset=calibration_dataset,
    batch_size=1, shuffle=True)

for batch in calibration_dataset:
    break
# batch = calibration_dataset

print(batch)
collate_fn = lambda x: {k:x[k].cuda() for k in batch}
inputs = { k:batch[k].cuda() for k in batch}

print("--------------------")
print(inputs)
outputs = model(**inputs)
print(outputs.sample.shape)
print("--------------------")

ppq_quant_ir = quantize_torch_model(
    model=model, calib_dataloader=calibration_dataset, input_shape=None, inputs=inputs, input_dtype=None,
    calib_steps=10, collate_fn=collate_fn, verbose=1,
    device=DEVICE, platform=PLATFORM, setting=quant_setting)