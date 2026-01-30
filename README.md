IMPORTANT: This repo is a fork of [SD-VLM](https://github.com/cpystan/SD-VLM), but rebased off of the [LLaVA](https://github.com/haotian-liu/LLaVA) and [Depth Anything V2](https://github.com/DepthAnything-Vision/Depth-Anything-V2) repos, so that the changes are clear.


# 👷SD-VLM: Spatial Measuring and Understanding with Depth-Encoded Vision-Language Models


[📢 [[Project Page](https://cpystan.github.io/SD_VLM_pages/)] [[Arxiv](https://arxiv.org/abs/2509.17664)]  [[Data](https://huggingface.co/datasets/cpystan/MSMU)] [[Model Zoo](https://huggingface.co/cpystan/SD-VLM-7B)] 



> **We are excited to announce that our paper is accepted by NeurIPS 2025!**

---

git clone https://github.com/thandal/LLaVA-SD-VLM.git
cd LLaVA-SD-VLM
```

2. Install Package
conda create -n sdvlm python=3.10 -y
conda activate sdvlm
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
```

### Quick Start With HuggingFace


```Python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from llava.mm_utils import tokenizer_image_token, process_images
from llava.constants import IMAGE_TOKEN_INDEX
import copy
from PIL import Image
import os
import torch

model_path = "cpystan/SD-VLM-7B"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_name=get_model_name_from_path(model_path),
    model_base=None,
    load_4bit=True,
    mm_vision_tower="openai/clip-vit-large-patch14-336"
)
model.gt_depth = False

prompt = "how big is the table?"
image_folder = "example_data"
image_file = "table_256.jpg"

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
ori_img = copy.deepcopy(image)
image_tensor = process_images([image], image_processor, model.config)[0]

temperature = 0.2
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor.unsqueeze(0).half().cuda(),
        image_sizes=[image.size],
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_p=None,
        num_beams=1,
        ori_imgs = [ori_img],
        max_new_tokens=1024,
        use_cache=True,)

response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
print(response)
```


## MSMU (Massive Spatial Measuring and Understanding) Dataset
For instruction tuning, please download the train.parquet of MSMU Dataset from [[HuggingFace](https://huggingface.co/datasets/cpystan/MSMU)]. 

For evaluation on MSMU-Bench, please download the test.parquet of MSMU Dataset from [[HuggingFace](https://huggingface.co/datasets/cpystan/MSMU)]. 

## Training
SD-VLM inherits the instruction-tuning pipeline of LLaVA, based on the well-established checkpoint of [LLaVA-1.5-7B](https://github.com/haotian-liu/LLaVA/). It requires relatively low resources for GPUs since SD-VLM can be trained with LoRA on 8 V100 GPUs. 

1. LoRA Finetuning (official setting)
```Shell
sh scripts/v1_5/finetune_task_lora.sh
```

2. Non-LoRA Finetuning 
```Shell
sh scripts/v1_5/finetune_task.sh
```

Some arguments in the script need to be modified:

- `--model_name_or_path`: path to the checkpoint of LLaVA-1.5-7B
- `--data_path`: path to the train set of MSMU
- `--vision_tower`: path to clip-vit-large-patch14-336
- `--depth_path`: path to depth_anything_v2_vitl


## Evaluation on MSMU-Bench

Step 1: Load the checkpoint and conduct inference on MSMU-Bench.
```Shell
python evaluation/inference.py --model-path 'SD-VLM-7B' --data 'MSMU' --vision-tower 'clip-vit-large-patch14-336'
```

Step 2: Load the json file saved from step 1 and use GPT-4-Turbo to get the evaluation response.
```Shell
python evaluation/llm_generate.py --path './preds.json' --key 'gpt_key'
```

Step 3: Load the json file saved from step 2 and calculate the final result.
```Shell
python evaluation/evaluate_final.py --path './preds_gpt.json' 
```

## Citation
If you find SD-VLM useful for your research and applications, please cite using this BibTeX:

```Shell
@misc{chen2025sdvlm,
      title={SD-VLM: Spatial Measuring and Understanding with Depth-Encoded Vision-Language Models}, 
      author={Pingyi Chen and Yujing Lou and Shen Cao and Jinhui Guo and Lubin Fan and Yue Wu and Lin Yang and Lizhuang Ma and Jieping Ye},
      year={2025},
      url={https://arxiv.org/abs/2509.17664}, 
}
```
