import argparse
import torch
import numpy as np
from diffusers import AutoModel, WanPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.hooks.group_offloading import apply_group_offloading
from diffusers.utils import export_to_video, load_image
from transformers import UMT5EncoderModel

class Text2Vedio:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        self.model_name = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        self.text_encoder = UMT5EncoderModel.from_pretrained(self.model_name, subfolder="text_encoder", torch_dtype=torch.bfloat16)
        self.vae = AutoModel.from_pretrained(self.model_name, subfolder="vae", torch_dtype=torch.float32)
        self.transformer = AutoModel.from_pretrained(self.model_name, subfolder="transformer", torch_dtype=torch.bfloat16)
        self.pipe = WanPipeline.from_pretrained(
            self.model_name,
            vae=self.vae,
            transformer=self.transformer,
            text_encoder=self.text_encoder,
            dtype=self.dtype,
            device=self.device,
        )

    def text_to_video(self, text):
        prompt = """
        The camera rushes from far to near in a low-angle shot, 
        revealing a white ferret on a log. It plays, leaps into the water, and emerges, as the camera zooms in 
        for a close-up. Water splashes berry bushes nearby, while moss, snow, and leaves blanket the ground. 
        Birch trees and a light blue sky frame the scene, with ferns in the foreground. Side lighting casts dynamic 
        shadows and warm highlights. Medium composition, front view, low angle, with depth of field.
        """
        negative_prompt = """
        Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, 
        low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, 
        misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards
        """

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=81,
            guidance_scale=5.0,
        ).frames[0]
        return output

    def save_video(self, output, output_path):
        export_to_video(output, "output.mp4", fps=16)
    
def main():
    argparser = argparse.ArgumentParser(description="Text to Video Generation")
    argparser.add_argument("--text", type=str, required=True, help="Input text prompt for video generation")
    argparser.add_argument("--output_path", type=str, default="output.mp4", help="Path to save the generated video")
    args = argparser.parse_args()

    t2v = Text2Vedio()
    video_frames = t2v.text_to_video(args.text)
    t2v.save_video(video_frames, args.output_path)
    print(f"Video saved to {args.output_path}")

if __name__ == "__main__":
    main()