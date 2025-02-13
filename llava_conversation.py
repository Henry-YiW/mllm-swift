import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Load the model in half-precision
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf", 
    torch_dtype=torch.float16, 
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# Get single image
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Prepare single conversation
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]

# Process the conversation
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# Process single image and prompt
inputs = processor(
    images=image, 
    text=prompt, 
    return_tensors="pt"
).to(model.device, torch.float16)

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=30, output_hidden_states=True, output_attentions=True)
output = processor.batch_decode(generate_ids, skip_special_tokens=True)
print(output[0])