
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/latent-consistency-model-colab/blob/main/latent_consistency_model_colab.ipynb) 

## 🔥 Local gradio Demos (Text-to-Image):

To run the model locally, you can download the "local_gradio" folder:
1. Install Pytorch (CUDA). MacOS system can download the "MPS" version of Pytorch. Please refer to: [https://pytorch.org](https://pytorch.org). Install [Intel Extension for Pytorch](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/) as well if you're using Intel GPUs.
2. Install the main library:
```
pip install diffusers transformers accelerate gradio==3.48.0 
```
3. Launch the gradio: (For MacOS users, need to set the device="mps" in app.py; For Intel GPU users, set `device="xpu"` in app.py)
```
python app.py
```

**LCM Model Download**: [LCM_Dreamshaper_v7](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7)


By distilling classifier-free guidance into the model's input, LCM can generate high-quality images in very short inference time. We compare the inference time at the setting of 768 x 768 resolution, CFG scale w=8, batchsize=4, using a A800 GPU. 



## Usage
We have official [**LCM Pipeline**](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/latent_consistency_models) and [**LCM Scheduler**](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_lcm.py) in 🧨 Diffusers library now! The older usages will be deprecated.

You can try out Latency Consistency Models directly on:
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/SimianLuo/Latent_Consistency_Model)

To run the model yourself, you can leverage the 🧨 Diffusers library:
1. Install the library:
```
pip install --upgrade diffusers  # make sure to use at least diffusers >= 0.22
pip install transformers accelerate
```

2. Run the model:
```py
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")

# To save GPU memory, torch.float16 can be used, but it may compromise image quality.
pipe.to(torch_device="cuda", torch_dtype=torch.float32)

prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"

# Can be set to 1~50 steps. LCM support fast inference even <= 4 steps. Recommend: 1~8 steps.
num_inference_steps = 4 

images = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=8.0, lcm_origin_steps=50, output_type="pil").images
```



