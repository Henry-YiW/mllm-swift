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
import transformers
from model.swift.utils import initialize_swift, reset_swift_mode, clone_past_key_values, sample, swift_verify
from model.swift.modeling_llava import LlavaForConditionalGeneration
from model.swift.modeling_llama import LlamaForCausalLM
from model.swift.kv_cache import initialize_past_key_values, KVCache  # KVCache initialization

logging.basicConfig(level=logging.INFO)

def main():
    model_id = "llava-hf/llava-1.5-7b-hf"
    max_new_tokens = 50
    logits_processor = None  # Replace with an actual logits processor if available

    # Load the model and force all parameters onto cuda:0.
    # model = LlavaForConditionalGeneration.from_pretrained(
    #     model_id,
    #     low_cpu_mem_usage=True,
    #     device_map={'': 'cuda:0'}  # Force the entire model onto cuda:0
    # ).to("cuda:0")
    
    model_correct = transformers.LlavaForConditionalGeneration.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        device_map={'': 'cuda:0'}  # Force the entire model onto cuda:0
    ).to("cuda:0")
    
    # Initialize the language model component.
    #model_correct.init_language_model()

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
    inputs = auto_processor(images=raw_image, text=prompt, return_tensors='pt').to("cuda:0")
    input_ids = inputs["input_ids"]
    #print the shape of the input_ids
    print("type of input_ids", type(input_ids))
    print("input_ids shape", input_ids.shape)
    pixel_values = inputs.get("pixel_values", None)
    if pixel_values is not None:
        pixel_values = pixel_values.to("cuda:0", torch.float16)
   
    input_len = input_ids.shape[1]
    cur_length = input_len  # (for bookkeeping if needed)

    
    with torch.inference_mode():
        outputs_correct, inputs_holder = model_correct(
                        **inputs,
                        past_key_values=None,
                        #position_ids=None,
                    )
        print("outputs_correct", len(outputs_correct[1][0]))
        #get the logits of the last token
        logits_correct = outputs_correct[0][:, -1, :]
        probabilities_correct = torch.nn.functional.softmax(logits_correct, dim=-1)
        print("probabilities_correct:", probabilities_correct)
        #get the max probability
        max_prob_correct = torch.max(probabilities_correct, dim=-1)
        print("max probability_correct:", max_prob_correct)
        print("max probability_correct index:", torch.argmax(probabilities_correct, dim=-1))
        print("outputs_correct", tokenizer.decode(torch.argmax(probabilities_correct, dim=-1)))
        #print("inputs_holder", inputs_holder)
    #do forward pass on the correct model
    # print the position id of inpus_holder
    #print("position_ids", inputs_holder["position_ids"])
if __name__ == "__main__":
    main()