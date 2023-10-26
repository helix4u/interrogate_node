from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    FieldDescriptions,
    Input,
    InputField,
    InvocationContext,
    OutputField,
    UIComponent,
    UIType,
    WithMetadata,
    WithWorkflow,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.primitives import StringOutput, ImageField
from PIL import Image
from clip_interrogator import Config, Interrogator
import torch

# Initialize Interrogator
config = Config()
config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
config.blip_offload = False if torch.cuda.is_available() else True
config.chunk_size = 2048
config.flavor_intermediate_count = 512
config.blip_num_beams = 64
ci = Interrogator(config)

@invocation(
    "clip_interrogator_node",
    title="Image to CLIP Prompt",
    tags=["image", "text", "CLIP" ,"BLIP", "interrogate"],
    category="image",
    version="1.0.0",
)
class CLIPInterrogatorInvocation(BaseInvocation):
    """Generate a prompt from an image using clip_interrogator."""
    
    # Inputs
    image: ImageField = InputField(default=None, description="The image for generating a prompt")
    best_max_flavors: int = InputField(default=4, description="Max flavors for 'best' mode")

    def invoke(self, context: InvocationContext) -> StringOutput:
        # Get PIL image
        image = context.services.images.get_pil_image(self.image.image_name)
        
        # Generate prompt using CLIP
        prompt = ci.interrogate(image, max_flavors=int(self.best_max_flavors))
        
        return StringOutput(value=prompt)

@invocation_output("clip_interrogator_node_output")
class CLIPInterrogatorInvocationOutput(BaseInvocationOutput):
    generated_prompt: str = OutputField(description="The generated prompt")
