# SAM2
from transformers import AutoTokenizer
import torch, sys, os, io
from PIL import Image
import numpy as np

module_paths = ["./", "/home/alfred/mm/EVF-SAM"]

for module_path in module_paths:
    sys.path.append(os.path.abspath(module_path))
    
from model.evf_sam2 import EvfSam2Model
from inference import sam_preprocess, beit3_preprocess

model_path = "/models/hf/evf-sam2"
tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="right",
        use_fast=False,
    )
kwargs = {
    "torch_dtype": torch.half,
}
            
model = EvfSam2Model.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
del model.visual_model.memory_encoder
del model.visual_model.memory_attention
model = model.cuda("cuda:3").eval()


def seg_image(image, prompt:str):
#async def pred_image(file:UploadFile=File(...), prompt:str=Form(...)):
    #contents = await file.read()
    #print(f"In Sam2: {type(image)}")
    img = Image.open(io.BytesIO(image))
    image_np =  np.array(img)


    original_size_list = [image_np.shape[:2]]
    image_beit = beit3_preprocess(image_np, 224).to(dtype=model.dtype, device=model.device)
    image_sam, resize_shape = sam_preprocess(image_np, model_type="sam2")
    image_sam = image_sam.to(dtype=model.dtype, device=model.device)
    
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device=model.device)
    
    # infer
    pred_mask = model.inference(
        image_sam.unsqueeze(0),
        image_beit.unsqueeze(0),
        input_ids,
        resize_list=[resize_shape],
        original_size_list=original_size_list,
    )
    pred_mask = pred_mask.detach().cpu().numpy()[0]
    pred_mask = pred_mask > 0
    
    visualization = image_np.copy()
    visualization[pred_mask] = (
        image_np * 0.5
        + pred_mask[:, :, None].astype(np.uint8) * np.array([50, 120, 220]) * 0.5
    )[pred_mask]
    return visualization/255.0, pred_mask.astype(np.float16)

def pred_video():
    return True