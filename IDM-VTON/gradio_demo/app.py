import sys
sys.path.append('./')
from PIL import Image
import gradio as gr
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler, AutoencoderKL
from typing import List

import torch
import os
from transformers import AutoTokenizer
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    mask[binary_mask] = 1
    mask = (mask * 255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask

base_path = 'yisol/IDM-VTON'
example_path = os.path.join(os.path.dirname(__file__), 'example')

# Initialize models
unet = UNet2DConditionModel.from_pretrained(
    base_path,
    subfolder="unet",
    torch_dtype=torch.float16,
)
unet.requires_grad_(False)
tokenizer_one = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer",
    use_fast=False,
)
tokenizer_two = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer_2",
    use_fast=False,
)
noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

text_encoder_one = CLIPTextModel.from_pretrained(
    base_path,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    base_path,
    subfolder="text_encoder_2",
    torch_dtype=torch.float16,
)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    base_path,
    subfolder="image_encoder",
    torch_dtype=torch.float16,
)
vae = AutoencoderKL.from_pretrained(base_path, subfolder="vae", torch_dtype=torch.float16)

UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    base_path,
    subfolder="unet_encoder",
    torch_dtype=torch.float16,
)

parsing_model = Parsing(0)
openpose_model = OpenPose(0)

# Set models to evaluation mode
UNet_Encoder.requires_grad_(False)
image_encoder.requires_grad_(False)
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)

# Define image transformation
tensor_transfrom = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

# Initialize the pipeline
pipe = TryonPipeline.from_pretrained(
    base_path,
    unet=unet,
    vae=vae,
    feature_extractor=CLIPImageProcessor(),
    text_encoder=text_encoder_one,
    text_encoder_2=text_encoder_two,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler,
    image_encoder=image_encoder,
    torch_dtype=torch.float16,
)
pipe.unet_encoder = UNet_Encoder

def plot_images(info_dict, output_path='output'):
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Function to convert tensor to PIL Image
    def tensor_to_pil(tensor, is_pose=False):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        if is_pose:
            tensor = (tensor + 1) / 2
        tensor = tensor.clamp(0, 1)
        tensor = (tensor * 255).byte()
        return Image.fromarray(tensor.permute(1, 2, 0).numpy())

    # Save pose image
    pose_img = tensor_to_pil(info_dict['pose_img'], is_pose=True)
    pose_img.save(os.path.join(output_path, 'pose_image.png'))

    # Save garment image
    cloth_img = tensor_to_pil(info_dict['cloth'])
    cloth_img.save(os.path.join(output_path, 'garment_image.png'))

    # Save mask image
    if isinstance(info_dict['mask_image'], Image.Image):
        mask_img = info_dict['mask_image']
    else:
        mask_img = Image.fromarray((info_dict['mask_image'].squeeze().cpu().numpy() * 255).astype(np.uint8))
    mask_img.save(os.path.join(output_path, 'mask_image.png'))

    # Save human image
    if isinstance(info_dict['image'], Image.Image):
        human_img = info_dict['image']
    else:
        human_img = Image.fromarray((info_dict['image'].squeeze().cpu().numpy() * 255).astype(np.uint8))
    human_img.save(os.path.join(output_path, 'human_image.png'))

    # Save IP adapter image
    if isinstance(info_dict['ip_adapter_image'], Image.Image):
        ip_adapter_img = info_dict['ip_adapter_image']
    else:
        ip_adapter_img = Image.fromarray((info_dict['ip_adapter_image'].squeeze().cpu().numpy() * 255).astype(np.uint8))
    ip_adapter_img.save(os.path.join(output_path, 'ip_adapter_image.png'))

    print(f"All images have been saved in the '{output_path}' directory.")

def start_tryon(human_img, uploaded_mask, garm_img, garment_des, denoise_steps, seed):
    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)

    if human_img is None:
        raise gr.Error("Please upload a human image.")
    if garm_img is None:
        raise gr.Error("Please upload a garment image.")

    garm_img = garm_img.convert("RGB").resize((768, 1024))
    human_img = human_img.convert("RGB").resize((768, 1024))

    # Use the uploaded mask directly if provided
    if uploaded_mask is not None:
        mask = uploaded_mask.convert("RGB").resize((768, 1024))
        mask_gray = mask  # Use the uploaded mask as-is for display
    else:
        # Generate mask only if not uploaded
        keypoints = openpose_model(human_img.resize((384, 512)))
        model_parse, _ = parsing_model(human_img.resize((384, 512)))
        mask, _ = get_mask_location('hd', "upper_body", model_parse, keypoints)
        mask = mask.resize((768, 1024))
        mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
        mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    pose_img = args.func(args, human_img_arg)    
    pose_img = pose_img[:,:,::-1]    
    pose_img = Image.fromarray(pose_img).resize((768, 1024))
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            prompt = "model is wearing " + garment_des
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            
            prompt = "a photo of " + garment_des
            (
                prompt_embeds_c,
                _,
                _,
                _,
            ) = pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=negative_prompt,
            )

            pose_img = tensor_transfrom(pose_img).unsqueeze(0).to(device, torch.float16)
            garm_tensor = tensor_transfrom(garm_img).unsqueeze(0).to(device, torch.float16)
            generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
            
            images = pipe(
                prompt_embeds=prompt_embeds.to(device, torch.float16),
                negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                num_inference_steps=denoise_steps,
                generator=generator,
                strength=1.0,
                pose_img=pose_img,
                text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                cloth=garm_tensor,
                mask_image=mask,
                image=human_img, 
                height=1024,
                width=768,
                ip_adapter_image=garm_img.resize((768, 1024)),
                guidance_scale=2.0,
            )[0]

        info_dict = {
            'pose_img': pose_img,
            'cloth': garm_tensor,
            'mask_image': mask,
            'image': human_img,
            'ip_adapter_image': garm_img.resize((768, 1024)) 
        }

        print('Pose img:', info_dict['pose_img'])
        print('\n')
        print('cloth:', info_dict['cloth'])
        print('\n')
        print('Mask img:', info_dict['mask_image'])
        print('\n')
        print('Image:', info_dict['image'])
        print('\n')
        print('ip adapetr image:', info_dict['ip_adapter_image'])

        plot_images(info_dict)

    return images[0], mask_gray

garm_list = os.listdir(os.path.join(example_path, "cloth"))
garm_list_path = [os.path.join(example_path, "cloth", garm) for garm in garm_list]

human_list = os.listdir(os.path.join(example_path, "human"))
human_list_path = [os.path.join(example_path, "human", human) for human in human_list]

image_blocks = gr.Blocks().queue()

with image_blocks as demo:
    gr.Markdown("## IDM-VTON 👕👔👚")
    gr.Markdown("Virtual Try-on with your image and garment image. Check out the [source codes](https://github.com/yisol/IDM-VTON) and the [model](https://huggingface.co/yisol/IDM-VTON)")
    with gr.Row():
        with gr.Column():
            human_img = gr.Image(sources='upload', type="pil", label='Human Image', interactive=True)
            uploaded_mask = gr.Image(sources='upload', type="pil", label='Upload Mask (optional)', visible=True)
            example = gr.Examples(
                inputs=human_img,
                examples=human_list_path
            )

        with gr.Column():
            garm_img = gr.Image(label="Garment", sources='upload', type="pil")
            prompt = gr.Textbox(placeholder="Description of garment ex) Short Sleeve Round Neck T-shirts", label="Garment Description", elem_id="prompt")
            example = gr.Examples(
                inputs=garm_img,
                examples=garm_list_path)
        with gr.Column():
            masked_img = gr.Image(label="Masked image output", elem_id="masked-img", show_share_button=False)
        with gr.Column():
            image_out = gr.Image(label="Output", elem_id="output-img", show_share_button=False)

    with gr.Column():
        try_button = gr.Button(value="Try-on")
        with gr.Accordion(label="Advanced Settings", open=False):
            with gr.Row():
                denoise_steps = gr.Number(label="Denoising Steps", minimum=20, maximum=40, value=30, step=1)
                seed = gr.Number(label="Seed", minimum=-1, maximum=2147483647, step=1, value=42)

    try_button.click(fn=start_tryon, 
                     inputs=[human_img, uploaded_mask, garm_img, prompt, denoise_steps, seed], 
                     outputs=[image_out, masked_img])

image_blocks.launch(share=True)