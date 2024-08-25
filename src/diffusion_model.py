import torch
from diffusers import UNet2DConditionModel, DDPMPipeline, DDPMScheduler

model = UNet2DConditionModel(
    sample_size=32,  # Example size
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 512),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
)

scheduler = DDPMScheduler(num_train_timesteps=1000)

pipeline = DDPMPipeline(unet=model, scheduler=scheduler)

conditioning_data = torch.randn(16, 3) 

print(pipeline(conditioning_data))