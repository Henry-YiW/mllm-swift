#!/usr/bin/env python3
"""
A minimal test script to exercise the initialize_swift function of the modified LlavaForConditionalGeneration model.
This script loads the model and processor, prepares inputs, initializes the KVCache, resets swift mode,
and then calls initialize_swift to obtain the swift logits, a sampled token, and its top1 probability.
"""

import argparse
from PIL import Image
import requests
import torch
import logging

from transformers import AutoTokenizer, AutoProcessor, LlavaProcessor
from bayes_opt import BayesianOptimization, UtilityFunction

# Import swift utilities and model components.
from model.swift.utils import initialize_swift, reset_swift_mode, clone_past_key_values, sample
from model.swift.modeling_llava import LlavaForConditionalGeneration
from model.swift.modeling_llama import LlamaForCausalLM
from model.swift.kv_cache import initialize_past_key_values, KVCache  # KVCache initialization

logging.basicConfig(level=logging.INFO)

def main():
    model_id = "llava-hf/llava-1.5-7b-hf"
    max_new_tokens = 50
    logits_processor = None  # Replace with an actual logits processor if available

    # Load the model and force all parameters onto cuda:0.
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        device_map={'': 'cuda:0'}  # Force the entire model onto cuda:0
    ).to("cuda:0")
    #print the skip layers of language model
    # print("_attn_skip_layer_id_set:", model.get_skip_layers())
    # print("_mlp_skip_layer_id_set:", model.get_skip_layers())
    # Initialize the language model component.
    model.init_language_model()

    # Load the processor to obtain the tokenizer.
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
    # Apply the chat template to format the input as expected by the model.
    prompt = auto_processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Load an image from a URL.
    image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    raw_image = Image.open(requests.get(image_file, stream=True).raw)

    # Process the inputs (both image and formatted text) into model inputs.
    inputs = auto_processor(images=raw_image, text=prompt, return_tensors='pt')
    input_ids = inputs["input_ids"].to("cuda:0")
    pixel_values = inputs.get("pixel_values", None)
    if pixel_values is not None:
        pixel_values = pixel_values.to("cuda:0", torch.float16)
    # ----- Initialize KVCache -----
    # initialize_past_key_values returns:
    #   past_key_values: a list (per-layer) of lists (key/value pairs) of KVCache objects,
    #   past_key_values_data: a list of preallocated cache tensors,
    #   current_length_data: a tensor tracking the current cache length.

    past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(model.model)
    model.past_key_values = past_key_values
    model.past_key_values_data = past_key_values_data
    model.current_length_data = current_length_data

    # ----- Prepare for Swift Initialization -----
    # Get the current input length.
    input_len = input_ids.shape[1]
    cur_length = input_len  # (for bookkeeping if needed)

    # Reset the swift mode before generating.
    reset_swift_mode(model)

    # ----- Call initialize_swift -----
    swift_logits, sample_token, top1_prob = initialize_swift(
         input_ids,
         model,
         max_new_tokens,
         past_key_values,
         past_key_values_data,
         current_length_data,
         logits_processor=logits_processor,
         pixel_values=pixel_values
    )
    # print("Swift logits:", swift_logits)
    # print("Sampled token:", sample_token)
    # print("Top1 probability:", top1_prob)

    # Additional sequential generation/testing can be added here if necessary.

if __name__ == "__main__":
    main()