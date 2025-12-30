#!/usr/bin/env python3
"""
Script to save a raw HuggingFace model to a local directory.
This can be used to create a "checkpoint 0" for tracking training progress.
"""
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def save_base_model(
    model_name: str,
    output_dir: str,
    trust_remote_code: bool = True,
):
    """
    Download and save a HuggingFace model to a local directory.
    
    Args:
        model_name: HuggingFace model identifier (e.g., "Qwen/Qwen3-8B-Base")
        output_dir: Target directory to save the model
        trust_remote_code: Whether to trust remote code in the model repo
    """
    print(f"Loading model from HuggingFace: {model_name}")
    print(f"Target directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if directory is empty (to avoid overwriting)
    if os.listdir(output_dir):
        print(f"Warning: Target directory {output_dir} is not empty.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Load and save config
    print("Loading config...")
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code
    )
    config.save_pretrained(output_dir)
    
    # Load and save tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code
    )
    tokenizer.save_pretrained(output_dir)
    
    # Load and save model
    print("Loading model (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype="auto",
    )
    
    print(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    
    print("Done! Model saved successfully.")
    print(f"Model location: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Save a raw HuggingFace model to a local directory"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-8B-Base",
        help="HuggingFace model identifier (default: Qwen/Qwen3-8B-Base)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data2/haikang/projects/verl_dynamics/checkpoints/converted/Qwen3-8B-Base/correct_dapo_aime_0",
        help="Target directory to save the model"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Whether to trust remote code in the model repo"
    )
    
    args = parser.parse_args()
    
    save_base_model(
        model_name=args.model_name,
        output_dir=args.output_dir,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()

