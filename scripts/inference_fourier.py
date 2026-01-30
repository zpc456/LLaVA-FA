#!/usr/bin/env python3
"""
Inference script for LLaVA-FA (Fourier Approximation)
Supports loading and running inference with Fourier-compressed models
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch
from PIL import Image
import requests
from transformers import AutoTokenizer
import json

# Add LLaVA-FA to path
sys.path.append(str(Path(__file__).parent.parent))

from llavafd.model import LlavaLlamaForCausalLM
from llavafd.fourier import inject_fourier_adapters, FourierAdapterConfig, BasisType
from llavafd.conversation import conv_templates, SeparatorStyle
from llavafd.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llavafd.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from llavafd.utils import disable_torch_init

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA-FA Inference")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the LLaVA-FA model")
    parser.add_argument("--fourier_adapters_path", type=str, default=None,
                       help="Path to Fourier adapters (if separate)")
    parser.add_argument("--model_base", type=str, default=None,
                       help="Path to base model")
    parser.add_argument("--conv_mode", type=str, default="llava_v1",
                       help="Conversation mode")
    
    # Input arguments
    parser.add_argument("--image_path", type=str, default=None,
                       help="Path to input image")
    parser.add_argument("--query", type=str, default="Describe this image.",
                       help="Input query/question")
    parser.add_argument("--input_file", type=str, default=None,
                       help="JSON file with multiple inputs")
    
    # Generation arguments
    parser.add_argument("--temperature", type=float, default=0.2,
                       help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling")
    parser.add_argument("--num_beams", type=int, default=1,
                       help="Number of beams for beam search")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum new tokens to generate")
    
    # Other arguments
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file for results")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    return parser.parse_args()


def load_image(image_path):
    """Load image from path or URL"""
    try:
        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        logger.error(f"Failed to load image from {image_path}: {e}")
        return None


def load_model_and_tokenizer(args):
    """Load model and tokenizer"""
    disable_torch_init()
    
    # Get model name
    model_name = get_model_name_from_path(args.model_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    
    # Load model
    if args.model_base is not None:
        # Load from base model
        model = LlavaLlamaForCausalLM.from_pretrained(
            args.model_base,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16
        )
        
        # Load fine-tuned weights
        mm_projector_weights = torch.load(
            os.path.join(args.model_path, 'mm_projector.bin'), 
            map_location='cpu'
        )
        mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
        model.load_state_dict(mm_projector_weights, strict=False)
    else:
        # Load complete model
        model = LlavaLlamaForCausalLM.from_pretrained(
            args.model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16
        )
    
    # Load Fourier adapters if specified
    if args.fourier_adapters_path:
        logger.info(f"Loading Fourier adapters from {args.fourier_adapters_path}")
        
        # Load adapter configuration
        checkpoint = torch.load(args.fourier_adapters_path, map_location='cpu')
        config = checkpoint.get('config', {})
        
        # Create Fourier adapter configuration
        fourier_config = FourierAdapterConfig(
            target_modules=config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"]),
            basis_type=BasisType(config.get('basis_type', 'dct')),
            compression_ratios=config.get('compression_ratios', {}),
            scaling=config.get('scaling', 1.0),
            dropout=0.0  # No dropout during inference
        )
        
        # Inject adapters
        model, injector = inject_fourier_adapters(model, fourier_config, verbose=True)
        
        # Load adapter weights
        injector.load_adapters(args.fourier_adapters_path, model)
    
    # Setup vision tower
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device='cuda', dtype=torch.float16)
    
    # Setup image processor
    image_processor = vision_tower.image_processor
    
    model.cuda()
    model.eval()
    
    return model, tokenizer, image_processor


def prepare_inputs(image, query, tokenizer, image_processor, conv_mode="llava_v1"):
    """Prepare inputs for the model"""
    # Setup conversation
    if "llama-2" in conv_mode:
        conv_mode = "llava_llama_2"
    elif "mistral" in conv_mode or "mixtral" in conv_mode:
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in conv_mode:
        conv_mode = "chatml_direct"
    elif "v1" in conv_mode:
        conv_mode = "llava_v1"
    else:
        conv_mode = "llava_v0"
    
    conv = conv_templates[conv_mode].copy()
    
    # Handle image token
    if image is not None:
        # Process image
        if hasattr(image_processor, 'preprocess'):
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            image_tensor = image_processor(image, return_tensors='pt')['pixel_values'][0]
        
        image_tensor = image_tensor.unsqueeze(0).half().cuda()
        
        # Add image token to query if not present
        if DEFAULT_IMAGE_TOKEN not in query:
            query = DEFAULT_IMAGE_TOKEN + '\n' + query
        
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Tokenize
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).cuda()
        
        return {
            'input_ids': input_ids,
            'images': image_tensor
        }
    else:
        # Text-only input
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
        
        return {
            'input_ids': input_ids,
            'images': None
        }


def generate_response(model, tokenizer, inputs, args):
    """Generate response from model"""
    # Setup stopping criteria
    stop_str = tokenizer.decode([tokenizer.eos_token_id])
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, inputs['input_ids'])
    
    # Generate
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=inputs['input_ids'],
            images=inputs['images'],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )
    
    # Decode response
    output_text = tokenizer.decode(
        output_ids[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    ).strip()
    
    return output_text


def run_single_inference(model, tokenizer, image_processor, args):
    """Run inference on single image-query pair"""
    # Load image
    image = None
    if args.image_path:
        image = load_image(args.image_path)
        if image is None:
            logger.error("Failed to load image")
            return None
    
    # Prepare inputs
    inputs = prepare_inputs(image, args.query, tokenizer, image_processor, args.conv_mode)
    
    # Generate response
    response = generate_response(model, tokenizer, inputs, args)
    
    result = {
        'query': args.query,
        'image_path': args.image_path,
        'response': response
    }
    
    return result


def run_batch_inference(model, tokenizer, image_processor, args):
    """Run inference on batch of inputs from file"""
    with open(args.input_file, 'r') as f:
        inputs_data = json.load(f)
    
    results = []
    
    for i, item in enumerate(inputs_data):
        logger.info(f"Processing item {i+1}/{len(inputs_data)}")
        
        # Load image
        image = None
        if 'image_path' in item:
            image = load_image(item['image_path'])
            if image is None:
                logger.warning(f"Failed to load image for item {i+1}")
                continue
        
        # Prepare inputs
        query = item.get('query', item.get('question', 'Describe this image.'))
        inputs = prepare_inputs(image, query, tokenizer, image_processor, args.conv_mode)
        
        # Generate response
        response = generate_response(model, tokenizer, inputs, args)
        
        result = {
            'id': item.get('id', i),
            'query': query,
            'image_path': item.get('image_path'),
            'response': response
        }
        
        results.append(result)
        
        if args.debug:
            print(f"Query: {query}")
            print(f"Response: {response}")
            print("-" * 50)
    
    return results


def main():
    args = parse_args()
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer, image_processor = load_model_and_tokenizer(args)
    logger.info("Model loaded successfully")
    
    # Run inference
    if args.input_file:
        # Batch inference
        results = run_batch_inference(model, tokenizer, image_processor, args)
    else:
        # Single inference
        result = run_single_inference(model, tokenizer, image_processor, args)
        results = [result] if result else []
    
    # Print results
    for result in results:
        if result:
            print("\n" + "="*50)
            print(f"Query: {result['query']}")
            if result.get('image_path'):
                print(f"Image: {result['image_path']}")
            print(f"Response: {result['response']}")
    
    # Save results if output file specified
    if args.output_file and results:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
