
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/latent-consistency-model-colab/blob/main/latent_consistency_model_colab.ipynb) 


<table class="center">
<tr>
  <td style="text-align:center;"><b>Output</b></td>
  <td style="text-align:center;" ><b>Output</b></td>
  <td style="text-align:center;" ><b>Output</b></td>
</tr>
  
<tr>
  <td><img src="https://github.com/user-attachments/assets/08677159-2ca2-4e66-8efb-0d135de6ffb5"></td>
  <td><img src="https://github.com/user-attachments/assets/e4d5c7a7-9f15-4c82-880e-4ff7e6da9adc"></td>
  <td><img src="https://github.com/user-attachments/assets/18210feb-094b-4d19-ac82-47d4db39848b"></td>

</tr>

<tr>
  <td width=25% style="text-align:center;color:gray;">"A group of adorable animals having a picnic in a sunny meadow, including a bunny, a squirrel, a fox, and a hedgehog, with a colorful blanket and baskets of fruit"</td>
  <td width=25% style="text-align:center;">"A beautifully manicured Victorian garden with blooming flowers, elegant statues, and a grand fountain in the center, under a bright blue sky"</td>
  <td width=25% style="text-align:center;">"A friendly robot companion walking alongside a child in a park, with blooming flowers and a beautiful sunset in the background"</td>
</tr>
     
<tr>
  <td><img src="https://github.com/user-attachments/assets/d0130a5f-3a09-4267-aa1a-8c7bc053210c"></td>
  <td><img src="https://github.com/user-attachments/assets/309beb07-cb65-4a44-92e3-ea70772f0576"></td>
  <td><img src="https://github.com/user-attachments/assets/b1d0d9d1-f06b-4dd1-93b0-7e9368522ea6"></td>
 
             
 
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"An old, enchanted library with towering bookshelves, floating books, and a cozy reading nook with a fireplace casting a warm glow"</td>
  <td width=25% style="text-align:center;">"A picturesque winter wonderland with snow-covered trees, a frozen lake, and a cozy cabin with smoke curling from the chimney"</td>
  <td width=25% style="text-align:center;">"A serene, mystical forest with bioluminescent plants glowing in various colors under a starry night sky."</td>
 </tr>

<tr>
  <td><img src="https://github.com/user-attachments/assets/3036a5b2-67bb-4185-84ba-ec28afa21385"></td>
  <td><img src="https://github.com/user-attachments/assets/553d08c9-763e-441d-ba68-ed64d8603aac"></td>   
  <td><img src="https://github.com/user-attachments/assets/d5fa9677-a20f-47b9-8f76-03ac5a1c031d"></td>

</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"A grand fantasy castle perched on a cliff, with turrets and spires reaching into the clouds, surrounded by a magical, swirling mist"</td>
  <td width=25% style="text-align:center;">"An astronaut standing on the surface of a distant planet, looking out at a stunning galaxy filled with stars, nebulae, and a nearby spaceship"</td>
  <td width=25% style="text-align:center;">"A vibrant alien landscape with strange, colorful plants, unusual rock formations, and a sky with multiple moons and planets visible"</td>

 
</tr>



</table>







## 🔥 (Text-to-Image):

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



