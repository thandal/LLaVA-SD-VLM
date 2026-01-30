# From https://huggingface.co/cpystan/SD-VLM-7B

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
import copy

model_path = "cpystan/SD-VLM-7B"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
ori_img = copy.deepcopy(image)
image_tensor = process_images([image], image_processor, model.config)[0]

with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor.unsqueeze(0).half().to(input_ids.device),
        image_sizes=[image.size],
        do_sample=True if temperature > 0 else False,
        temperature=0.2,
        top_p=None,
        num_beams=1,
        ori_imgs = [ori_img],
        max_new_tokens=1024,
        use_cache=True,)
response= tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

