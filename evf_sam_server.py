import argparse
import litserve as ls
import cv2
import numpy as np
from io import BytesIO

import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.device_count())  # Should be >0
print(torch.cuda.get_device_name(0))  # Should show your GPU model
breakpoint()

import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BitsAndBytesConfig

from fastapi import FastAPI, Response, UploadFile, Form
from fastapi.responses import JSONResponse
import base64
from PIL import Image

import sys
sys.path.append('/home/koyo/EVF-SAM')
from model.segment_anything.utils.transforms import ResizeLongestSide
# from inference_realtime_cv2 import init_models, beit3_preprocess, sam_preprocess



class EVFSAMAPI(ls.LitAPI):
    def setup(self, device) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Use float16 for the entire notebook to optimize inference speed
        torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()
        # Enable TF32 for Ampere GPUs to speed up matrix multiplications and convolution operations
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        print("Initialise model")
        self.tokenizer, self.model = self.init_models("YxZhang/evf-sam2", "fp16", False, False, "sam2")

    def decode_request(self, request) -> dict:
        # Decode the incoming request to extract the video frame and prompt
        text_prompt = request.get("text_prompt", "")
        # Extract image file
        image_file: UploadFile = request.get("image")
        if image_file is None:
            raise ValueError("No image file provided in the request.")
        image_bytes = image_file.file.read()

        # Debug log for image bytes
        if not image_bytes:
            raise ValueError("No data found in the image file.")

        print(f"Image bytes size: {len(image_bytes)} bytes")

        return {
            "image_bytes": image_bytes,
            "text_prompt": text_prompt,
        }

    def predict(self, inputs: dict) -> dict:
        try:
            image_pil = Image.open(BytesIO(inputs["image_bytes"])).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid image data: {e}")
        
        prompt = inputs["text_prompt"]
        
        # Convert PIL image to NumPy array
        image_np = np.array(image_pil)

        # OpenCV expects images in BGR format, so converting to RGB first is common
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Now you can get the original size (height, width)
        original_size_list = [image_np.shape[:2]]  # (height, width)

        # Preprocess for BEIT-3 and SAM
        image_beit = self.beit3_preprocess(image_np, img_size=224).to(dtype=self.model.dtype, device=self.device)
        image_sam, resize_shape = self.sam_preprocess(image_np, model_type="sam2")
        image_sam = image_sam.to(dtype=self.model.dtype, device=self.device)

        # Tokenize the prompt
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(device=self.device)

        # Run inference
        pred_mask = self.model.inference(
            image_sam.unsqueeze(0),
            image_beit.unsqueeze(0),
            input_ids,
            resize_list=[resize_shape],
            original_size_list=original_size_list,
        )

        pred_mask = torch.sigmoid(pred_mask)  # Convert logits to probabilities
        pred_mask = pred_mask > 0.5  # Apply threshold to get binary mask
        pred_mask = pred_mask.detach().cpu().numpy()[0]  # Convert to NumPy for further processing

        # Create an overlay on the original image
        overlay = image_np.copy()
        overlay[pred_mask] = (
            image_np * 0.5
            + pred_mask[:, :, None].astype(np.uint8) * np.array([50, 120, 220]) * 0.5
        )[pred_mask]
    
        # Convert overlay back to BGR for display
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        bb_image = image_np.copy()
        # Compute the bounding box from the prediction mask
        y, x = np.where(pred_mask)  # Get coordinates where the mask is True
        if len(y) > 0 and len(x) > 0:
            xmin, ymin, xmax, ymax = min(x), min(y), max(x), max(y)
            top_left = (xmin, ymin)  # Top-left corner of the bounding box
            bottom_right = (xmax, ymax)  # Bottom-right corner of the bounding box

            # Draw the bounding box directly on the original frame (in BGR format)
            cv2.rectangle(bb_image, top_left, bottom_right, (0, 255, 0), 2)  # Green box

        # # Create a black-and-white mask for display
        bw_mask = np.zeros_like(image_np, dtype=np.uint8)
        bw_mask[pred_mask] = 255
        # print("bw_mask: ", bw_mask)
        # print(f"bw_mask shape: {bw_mask.shape}, unique values: {np.unique(bw_mask)}")

        # print("xmin: ", xmin)
        # print("ymin: ", ymin)
        # print("xmax: ", xmax)
        # print("ymax: ", ymax)

        return {
            "segmentation_image": overlay, 
            "bounding_box_image": bb_image, 
            "mask_image": bw_mask, 
            "xmin": int(xmin), 
            "ymin": int(ymin), 
            "xmax": int(xmax), 
            "ymax": int(ymax)}
        # return {"segmentation_image": overlay, "bounding_box_image": bb_image}

    def encode_response(self, output: dict) -> Response:
        try:
            segmentation_image = output["segmentation_image"]
            segmentation_image_data = self.convert_image(segmentation_image)

            bounding_box_image = output["bounding_box_image"]
            bounding_box_image_data = self.convert_image(bounding_box_image)

            mask_image = output.get("mask_image", None)  # Check if mask_image exists
            if mask_image is None:
                raise ValueError("mask_image is missing from the output dictionary.")
            mask_image_data = self.convert_image(mask_image)
            
            response = {
                "segmentation_image": segmentation_image_data,
                "bounding_box_image": bounding_box_image_data,
                "mask_image": mask_image_data,
                "xmin": output["xmin"],
                "ymin": output["ymin"],
                "xmax": output["xmax"],
                "ymax": output["ymax"],
            }

            return JSONResponse(content=response)

        except StopIteration:
            raise ValueError("No output generated by the prediction.")

        

    def convert_image(self, image):
        # print("image: ", image)
        # output_image = Image.fromarray(np.uint8(output_image)).convert("RGB")
        
        # Check if the input is a NumPy array and convert it to a PIL image
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if image.shape[-1] == 3:  # Check if it has 3 channels
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        # Ensure the image is a byte representation (for PIL Image objects)
        if isinstance(image, Image.Image):  # Check if image is PIL.Image
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        elif isinstance(image, bytes):  # If it's already a byte object
            image_data = base64.b64encode(image).decode('utf-8')
        else:
            raise ValueError("Invalid image format.")
        
        return image_data


    def sam_preprocess(
        self,
        x: np.ndarray,
        pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
        pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
        img_size=1024,
        model_type="ori") -> torch.Tensor:
        '''
        preprocess of Segment Anything Model, including scaling, normalization and padding.  
        preprocess differs between SAM and Effi-SAM, where Effi-SAM use no padding.
        input: ndarray
        output: torch.Tensor
        '''
        assert img_size==1024, \
            "both SAM and Effi-SAM receive images of size 1024^2, don't change this setting unless you're sure that your employed model works well with another size."
        
        # Normalize colors
        if model_type=="ori":
            x = ResizeLongestSide(img_size).apply_image(x)
            h, w = resize_shape = x.shape[:2]
            x = torch.from_numpy(x).permute(2,0,1).contiguous()
            x = (x - pixel_mean) / pixel_std
            # Pad
            padh = img_size - h
            padw = img_size - w
            x = F.pad(x, (0, padw, 0, padh))
        else:
            x = torch.from_numpy(x).permute(2,0,1).contiguous()
            x = F.interpolate(x.unsqueeze(0), (img_size, img_size), mode="bilinear", align_corners=False).squeeze(0)
            x = (x - pixel_mean) / pixel_std
            resize_shape = None
        
        return x, resize_shape

    def beit3_preprocess(self, x: np.ndarray, img_size=224) -> torch.Tensor:
        '''
        preprocess for BEIT-3 model.
        input: ndarray
        output: torch.Tensor
        '''
        beit_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC, antialias=None), 
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        return beit_preprocess(x)

    # version, precision, load_in_4bit, load_in_8bit, model_type
    def init_models(self, version, precision, load_in_4bit, load_in_8bit, model_type):
        tokenizer = AutoTokenizer.from_pretrained(
            version,
            padding_side="right",
            use_fast=False,
        )

        torch_dtype = torch.float32
        if precision == "bf16":
            torch_dtype = torch.bfloat16
        elif precision == "fp16":
            torch_dtype = torch.half

        kwargs = {"torch_dtype": torch_dtype}
        if load_in_4bit:
            kwargs.update(
                {
                    "torch_dtype": torch.half,
                    "quantization_config": BitsAndBytesConfig(
                        llm_int8_skip_modules=["visual_model"],
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    ),
                }
            )
        elif load_in_8bit:
            kwargs.update(
                {
                    "torch_dtype": torch.half,
                    "quantization_config": BitsAndBytesConfig(
                        llm_int8_skip_modules=["visual_model"],
                        load_in_8bit=True,
                    ),
                }
            )

        if model_type=="ori":
            from model.evf_sam import EvfSamModel
            model = EvfSamModel.from_pretrained(
                version, low_cpu_mem_usage=True, **kwargs
            )
        elif model_type=="effi":
            from model.evf_effisam import EvfEffiSamModel
            model = EvfEffiSamModel.from_pretrained(
                version, low_cpu_mem_usage=True, **kwargs
            )
        elif model_type=="sam2":
            from model.evf_sam2 import EvfSam2Model
            model = EvfSam2Model.from_pretrained(
                version, low_cpu_mem_usage=False, **kwargs
            )
        
        # print("Model: ", model)
        # breakpoint()

        if (not load_in_4bit) and (not load_in_8bit):
            model = model.cuda()
        model.eval()

        return tokenizer, model


# lit_api = EVFSAM_API()
# server = ls.LitServer(lit_api)

# # Define the POST endpoint for prediction
# @app.post("/predict")
# async def predict(
#     text_prompt: str = Form(...),
#     image: UploadFile = Form(...)
# ):
#     print('predict endpoint')
#     inputs = {
#         "text_prompt": text_prompt,
#         "image_bytes": await image.read(),
#     }

#     # Use LangSAMAPI to make a prediction
#     output = lit_api.predict(inputs)
#     return lit_api.encode_response(output)


if __name__ == "__main__":
    # import torch
    # import sys
    # print('__Python VERSION:', sys.version)
    # print('__pyTorch VERSION:', torch.__version__)
    # print('__CUDA VERSION')
    # from subprocess import call
    # # call(["nvcc", "--version"]) does not work 
    # print('__CUDNN VERSION:', torch.backends.cudnn.version())
    # print('__Number CUDA Devices:', torch.cuda.device_count())
    # print('__Devices')
    # call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    # print('Active CUDA Device: GPU', torch.cuda.current_device())
    # print ('Available devices ', torch.cuda.device_count())
    # print ('Current cuda device ', torch.cuda.current_device())
    # print(f"Starting LitServe and Gradio server on port {PORT}...")

    # server.run(port=PORT, log_level="trace")
    api = EVFSAMAPI()
    server = ls.LitServer(api, accelerator="auto", max_batch_size=1)
    server.run(port=8000, log_level="trace")
