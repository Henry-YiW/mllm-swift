#!/usr/bin/env python3
"""
A minimal test script to exercise the model.generate method of the modified modeling_llava.
This script initializes a LlavaForConditionalGeneration model, sets up a processor and input,
and calls model.generate to produce an output.
"""

import argparse
from PIL import Image
import requests
import torch
import logging
from fastchat.utils import str_to_torch_dtype

from evaluation_llama.eval import run_eval

from transformers import AutoTokenizer, AutoProcessor, LlavaProcessor
from bayes_opt import BayesianOptimization, UtilityFunction

from model.swift.utils import *
from model.swift.modeling_llava import LlavaForConditionalGeneration
from model.swift.modeling_llama import LlamaForCausalLM
from model.swift.kv_cache import initialize_past_key_values, KVCache  # KVCache definitions are here

logging.basicConfig(level=logging.INFO)

def main():
    model_id = "llava-hf/llava-1.5-7b-hf"
    
    # Load the model and force all parameters to be on cuda:0.
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        device_map={'': 'cuda:0'}  # Force the entire model onto cuda:0
    ).to("cuda:0")
    
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
                {"type": "image"},
            ],
        },
    ]
    # Apply the chat template to format the input as expected by the model.
    prompt = auto_processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Load an image from a URL.
    image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    raw_image = Image.open(requests.get(image_file, stream=True).raw)

    # Process the inputs (both image and formatted text) into model inputs.
    inputs = auto_processor(images=raw_image, text=prompt, return_tensors='pt')
    if "input_ids" in inputs:
        inputs["input_ids"] = inputs["input_ids"].to("cuda:0")
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to("cuda:0", torch.float16)

    # ----- Initialize KVCache -----
    # This function returns three objects:
    #   - past_key_values: a list (per-layer) of lists (key/value pair) of KVCache objects.
    #   - past_key_values_data_list: a list of raw tensors preallocated for caching.
    #   - current_length_data: a tensor tracking current cache length (kept on CPU).
    (
        past_key_values,
        past_key_values_data,
        current_length_data,
    ) = initialize_past_key_values(model.model)
    model.past_key_values = past_key_values
    model.past_key_values_data = past_key_values_data #shape [64,1,32,4096,128]
    model.current_length_data = current_length_data
    # past_key_values_data = torch.zeros(
    #     startnum * 2,
    #     batch_size,
    #     config.num_key_value_heads,
    #     config.max_position_embeddings,
    #     config.hidden_size // config.num_attention_heads,
    #     device=startdevice,
    #     dtype=model.dtype,
    # )
    with torch.inference_mode():
        outputs, logits = swift_verify(model, inputs, past_key_values=past_key_values, pixel_values=inputs["pixel_values"])
    generated_ids = inputs["input_ids"]
    # # Initial KVCache update: only append the new key/value tokens from outputs.
    # new_past = outputs.past_key_values 
    # for layer_idx, kv_pair in enumerate(new_past):
    #     for index, key_or_value in enumerate(kv_pair):
    #         model.past_key_values[layer_idx][index] = KVCache(key_or_value, outputs.past_key_values[layer_idx][index].shape[2])

    # ----- Sequential generation loop -----
    for _ in range(max_new_tokens):
        # Prepare input: only the last generated token.
        next_input = generated_ids[:, -1:].clone()

        outputs = model(
            input_ids=next_input,
            past_key_values=past_key_values,  # Use our KVCache objects.
            use_cache=True,
            return_dict=True
        )
        new_past = outputs.past_key_values
        # Update our KVCache objects in place with the new token's representations
        # new_past = outputs.past_key_values
        # for layer_idx, kv_pair in enumerate(new_past):
        #     for j in range(2):
        #         # Because outputs.past_key_values includes the full history,
        #         # we only need the last token.
        #         incremental = new_past[layer_idx][j][..., -1:, :]
        #         model.past_key_values[layer_idx][j].cat(incremental, dim=2)

        # Extract logits of the new token.
        logits = outputs.logits
        
        # Greedily select the token with the highest probability.
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        
        # Append the new token to the generated sequence.
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        
        # Stop generation if EOS token is produced.
        if next_token.item() == tokenizer.eos_token_id:
            break

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("Sequential Generation with KV Cache:\n", generated_text)

if __name__ == "__main__":
    main()