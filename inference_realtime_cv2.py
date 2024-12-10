import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BitsAndBytesConfig

import sys
sys.path.append('/home/koyo/EVF-SAM')
from model.segment_anything.utils.transforms import ResizeLongestSide



def parse_args(args):
    parser = argparse.ArgumentParser(description="EVF infer")
    parser.add_argument("--version", required=True)
    parser.add_argument("--vis_save_path", default="./infer", type=str)
    parser.add_argument(
        "--precision",
        default="fp16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=224, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)

    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--model_type", default="ori", choices=["ori", "effi", "sam2"])
    parser.add_argument("--image_path", type=str, default="assets/zebra.jpg")
    parser.add_argument("--prompt", type=str, default="zebra top left")
    
    return parser.parse_args(args)


def sam_preprocess(
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

def beit3_preprocess(x: np.ndarray, img_size=224) -> torch.Tensor:
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

def init_models(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        padding_side="right",
        use_fast=False,
    )

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
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
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    if args.model_type=="ori":
        from model.evf_sam import EvfSamModel
        model = EvfSamModel.from_pretrained(
            args.version, low_cpu_mem_usage=True, **kwargs
        )
    elif args.model_type=="effi":
        from model.evf_effisam import EvfEffiSamModel
        model = EvfEffiSamModel.from_pretrained(
            args.version, low_cpu_mem_usage=True, **kwargs
        )
    elif args.model_type=="sam2":
        from model.evf_sam2 import EvfSam2Model
        model = EvfSam2Model.from_pretrained(
            args.version, low_cpu_mem_usage=True, **kwargs
        )

    if (not args.load_in_4bit) and (not args.load_in_8bit):
        model = model.cuda()
    model.eval()

    return tokenizer, model

def main(args):
    # Parse command-line arguments
    args = parse_args(args)

    # Use float16 for the entire notebook to optimize inference speed
    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

    # Enable TF32 for Ampere GPUs to speed up matrix multiplications and convolution operations
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Use the prompt from the command-line argument
    prompt = args.prompt

    # Initialize the model and tokenizer
    tokenizer, model = init_models(args)

    # Initialize video capture (0 = default webcam, or you can provide a video file path)
    cap = cv2.VideoCapture(0)  # Change 0 to the path of a video file if needed

    # Check if the video stream has opened successfully
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()

    print("Start streaming loop ....")
    # Start a loop to continuously capture frames from the video stream
    while True:
        ret, frame = cap.read()  # Read a frame from the video stream

        # If frame reading fails, break out of the loop
        if not ret:
            print("Error: Unable to read frame.")
            break

        # Convert the captured frame from BGR to RGB format for model input
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Save the original frame size for later use in resizing and visualization
        original_size_list = [frame_rgb.shape[:2]]

        # Preprocess the frame for the BEIT-3 model
        image_beit = beit3_preprocess(frame_rgb, args.image_size).to(
            dtype=model.dtype, device=model.device
        )

        # Preprocess the frame for the Segment Anything Model (SAM)
        image_sam, resize_shape = sam_preprocess(
            frame_rgb, model_type=args.model_type
        )
        image_sam = image_sam.to(dtype=model.dtype, device=model.device)

        # Tokenize the prompt to create input IDs for the model
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device=model.device)

        print("Running inference .....")
        # Run inference with the preprocessed frames and prompt
        pred_mask = model.inference(
            image_sam.unsqueeze(0),
            image_beit.unsqueeze(0),
            input_ids,
            resize_list=[resize_shape],
            original_size_list=original_size_list,
        )

        # Detach the prediction mask and convert it to a binary mask
        pred_mask = pred_mask.detach().cpu().numpy()[0]
        # print("pred_mask numpy: ", pred_mask)
        pred_mask = pred_mask > 0
        # print("pred_mask bool: ", pred_mask)


        print("Creating an overlay image ....")
        # Create an overlay image by applying the prediction mask to the original frame
        overlay = frame_rgb.copy()
        overlay[pred_mask] = (
            frame_rgb * 0.5
            + pred_mask[:, :, None].astype(np.uint8) * np.array([50, 120, 220]) * 0.5
        )[pred_mask]
        
        # Convert the overlay back to BGR format for OpenCV display
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        # Compute the bounding box from the prediction mask
        y, x = np.where(pred_mask)  # Get coordinates where the mask is True
        if len(y) > 0 and len(x) > 0:
            top_left = (min(x), min(y))  # Top-left corner of the bounding box
            bottom_right = (max(x), max(y))  # Bottom-right corner of the bounding box

            # Draw the bounding box directly on the original frame (in BGR format)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)  # Green box
        
        # Create a black-and-white mask for display
        bw_mask = np.zeros_like(frame, dtype=np.uint8)
        bw_mask[pred_mask] = 255  # Set predicted mask areas to white
        # print("bw_mask after bw_mask[pred_mask]=255: ", bw_mask)

        # print(f"Frame shape: {overlay.shape}") 
        print("Streaming ...")
        # Display the processed frame in a window
        # cv2.imshow("Real-Time Inference", overlay)
        # Display the processed frame with bounding box
        # cv2.imshow("Real-Time Inference", frame)
        # Display the black-and-white mask
        cv2.imshow("Predicted Mask", bw_mask)


        # Check if the user presses the 'q' key to quit the stream
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Call the main function with command-line arguments
    main(sys.argv[1:])
