#!/usr/bin/env python3
"""
A minimal test script to exercise the swift_verify function of the modified LlavaForConditionalGeneration model.
This script loads the model and processor, prepares inputs (including text and image), resets swift mode,
and then calls swift_verify to obtain the swift logits as well as the internal inputs_holder.
The inputs_holder is saved (pickled) to a file for later inspection.
"""

import argparse
from PIL import Image
import requests
import torch
import logging
import pickle

from transformers import AutoTokenizer, AutoProcessor, LlavaProcessor
import transformers

from model.swift.utils import swift_verify, reset_swift_mode
from model.swift.modeling_llava import LlavaForConditionalGeneration

logging.basicConfig(level=logging.INFO)

def main():
    model_id = "llava-hf/llava-1.5-7b-hf"
    max_new_tokens = 50

    # Load the Swift-modified model and push it to cuda:0.
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        device_map={'': 'cuda:0'}
    ).to("cuda:0")
    
    # Initialize the language model component.
    model.init_language_model()

    # Load the processor to get the tokenizer.
    auto_processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = auto_processor.tokenizer

    # Define a conversation template.
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What are these?"},
                {"type": "image"}
            ]
        }
    ]
    # Apply the chat template to format the text input.
    prompt = auto_processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Load an image from a URL.
    image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    raw_image = Image.open(requests.get(image_file, stream=True).raw)

    # Process inputs (both image and text) into model tensors.
    inputs = auto_processor(images=raw_image, text=prompt, return_tensors='pt').to("cuda:0")
    input_ids = inputs["input_ids"]
    pixel_values = inputs.get("pixel_values", None)
    if pixel_values is not None:
        pixel_values = pixel_values.to("cuda:0", torch.float16)

    # Reset swift mode before running verification.
    reset_swift_mode(model)

    with torch.inference_mode():
        # Call swift_verify.
        # Note: Adjust the arguments if your swift_verify API requires different parameters.
        outputs, inputs_holder = model(
            **inputs,
            past_key_values=None,
            #position_ids=None,
            return_raw=True
        )

        # Compute softmax probabilities on the last token's logits.
        logits = outputs[0]
        probabilities = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
        max_prob, _ = torch.max(probabilities, dim=-1)
        max_prob_index = torch.argmax(probabilities, dim=-1)

        print("Swift verify max probability:", max_prob)
        print("Swift verify predicted token index:", max_prob_index)
        print("Swift verify predicted token text:", tokenizer.decode(max_prob_index))

    # Save (pickle) the inputs_holder to a file.
    with open("swift_inputs_holder.pkl", "wb") as f:
        pickle.dump(inputs_holder, f)
    print("Saved swift inputs_holder to swift_inputs_holder.pkl")

if __name__ == "__main__":
    main()