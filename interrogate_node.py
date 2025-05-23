from typing import Literal

import torch
from clip_interrogator import Config, Interrogator

from invokeai.invocation_api import (
    BaseInvocation,
    ImageField,
    InputField,
    InvocationContext,
    invocation,
)
from invokeai.app.invocations.primitives import StringOutput

MODES = Literal["best", "classic", "fast", "negative"]
CLIP_MODELS = Literal[
    "ViT-L-14/openai", "ViT-H-14/laion2b_s32b_b79k", "ViT-bigG-14/laion2b_s39b_b160k"
]
CAPTION_MODELS = Literal[
    "blip-base", "blip-large", "blip2-2.7b", "blip2-flan-t5-xl", "git-large-coco"
]

ci = None

@invocation(
    "clip_interrogator_node",
    title="Image to CLIP Prompt",
    tags=["image", "text", "CLIP", "BLIP", "interrogate"],
    category="image",
    version="1.0.2",
)
class CLIPInterrogatorInvocation(BaseInvocation):
    """Generate a prompt from an image using clip_interrogator."""

    image: ImageField = InputField(
        description="The image for generating a prompt"
    )
    best_max_flavors: int = InputField(
        default=32, description="Max flavors for 'best' mode"
    )
    mode: MODES = InputField(default="best", description="Mode")
    clip_model: CLIP_MODELS = InputField(
        default="ViT-L-14/openai",
        description="Clip model. SD1:ViT-L , SD2:ViT-H, SDXL:ViT-L or ViT-bigG",
    )
    caption_model: CAPTION_MODELS = InputField(
        default="blip-large", description="Caption Model"
    )
    low_vram: bool = InputField(default=False, description="Low VRAM mode.")

    def invoke(self, context: InvocationContext) -> StringOutput:
        global ci

        if ci is None:
            config = Config()
            config.clip_model_name = self.clip_model
            config.caption_model_name = self.caption_model
            ci = Interrogator(config)

        image = context.images.get_pil(self.image.image_name).convert("RGB")

        if self.low_vram:
            ci.config.caption_offload = True
            ci.config.clip_offload = True
            ci.config.chunk_size = 1024
            ci.config.flavor_intermediate_count = 1024
        else:
            offload = not torch.cuda.is_available()
            ci.config.caption_offload = offload
            ci.config.clip_offload = offload
            ci.config.chunk_size = 2048
            ci.config.flavor_intermediate_count = 2048

        if self.clip_model != ci.config.clip_model_name:
            ci.config.clip_model_name = self.clip_model
            ci.load_clip_model()

        if self.caption_model != ci.config.caption_model_name:
            ci.config.caption_model_name = self.caption_model
            ci.load_caption_model()

        if self.mode == "best":
            prompt = ci.interrogate(image, max_flavors=int(self.best_max_flavors))
        elif self.mode == "classic":
            prompt = ci.interrogate_classic(image)
        elif self.mode == "fast":
            prompt = ci.interrogate_fast(image)
        elif self.mode == "negative":
            prompt = ci.interrogate_negative(image)

        # Clean up
        ci.clip_model = None
        ci.blip_model = None
        torch.cuda.empty_cache()
        ci = None

        return StringOutput(value=prompt)
