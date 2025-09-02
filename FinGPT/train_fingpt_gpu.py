#!/usr/bin/env python3
"""
FinGPT GPU Training Script - Optimized for Maximum GPU Performance
Handles all common issues: paths, modules, GPU memory, and storage
"""

import os
import sys
import json
import torch
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Fix import paths - add FinGPT directories to path
current_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "fingpt"))
sys.path.insert(0, str(current_dir / "fingpt" / "FinGPT_Forecaster"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FinGPTTrainer:
    """Main trainer class for FinGPT with GPU optimization"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize trainer with configuration"""
        self.config = self.load_config(config_path)
        self.device = self.setup_device()
        self.setup_directories()
        
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load training configuration"""
        default_config = {
            "model": {
                "base_model": "meta-llama/Llama-2-7b-hf",
                "model_path": None,  # Will be set based on environment
                "use_lora": True,
                "lora_config": {
                    "r": 8,
                    "lora_alpha": 32,
                    "target_modules": ["q_proj", "v_proj"],
                    "lora_dropout": 0.05,
                }
            },
            "training": {
                "batch_size": 4,
                "gradient_accumulation_steps": 4,
                "learning_rate": 2e-4,
                "num_epochs": 3,
                "warmup_ratio": 0.03,
                "max_length": 2048,
                "fp16": True,
                "gradient_checkpointing": True,
                "save_steps": 100,
                "eval_steps": 100,
                "logging_steps": 10,
            },
            "data": {
                "dataset": "fingpt-forecaster",
                "data_path": "./data",
                "train_split": 0.9,
                "max_samples": None,  # Set to limit samples for testing
            },
            "gpu": {
                "max_memory_mb": None,  # Auto-detect
                "mixed_precision": True,
                "optimize_memory": True,
                "use_deepspeed": False,
                "num_gpus": 1,
            },
            "paths": {
                "output_dir": "./output",
                "checkpoint_dir": "./checkpoints",
                "cache_dir": "./cache",
                "logs_dir": "./logs",
            }
        }
        
        # Load custom config if provided
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                # Deep merge configs
                self._merge_configs(default_config, custom_config)
                
        return default_config
    
    def _merge_configs(self, base: Dict, custom: Dict) -> None:
        """Merge custom config into base config"""
        for key, value in custom.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def setup_device(self) -> torch.device:
        """Setup and optimize GPU device"""
        if not torch.cuda.is_available():
            logger.warning("No GPU available, using CPU (not recommended)")
            return torch.device("cpu")
        
        # Get GPU info
        gpu_count = torch.cuda.device_count()
        logger.info(f"Found {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name}, Memory: {props.total_memory / 1024**3:.2f} GB")
        
        # Set primary device
        device = torch.device("cuda:0")
        
        # Optimize GPU settings
        if self.config["gpu"]["optimize_memory"]:
            # Enable memory efficient attention
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            
            # Clear cache
            torch.cuda.empty_cache()
            
        return device
    
    def setup_directories(self) -> None:
        """Create necessary directories"""
        for key, path in self.config["paths"].items():
            Path(path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path}")
    
    def check_dependencies(self) -> bool:
        """Check and install missing dependencies"""
        required_packages = [
            "transformers>=4.35.0",
            "peft>=0.5.0",
            "datasets>=2.14.0",
            "accelerate>=0.24.0",
            "bitsandbytes>=0.41.0",
            "sentencepiece",
            "protobuf",
            "wandb",
            "tensorboard",
        ]
        
        missing = []
        for package in required_packages:
            package_name = package.split(">=")[0]
            try:
                __import__(package_name)
            except ImportError:
                missing.append(package)
        
        if missing:
            logger.warning(f"Missing packages: {missing}")
            logger.info("Installing missing packages...")
            import subprocess
            for package in missing:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            logger.info("Dependencies installed successfully")
            
        return True
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with GPU optimization"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        
        logger.info(f"Loading model: {self.config['model']['base_model']}")
        
        # Model loading arguments
        model_kwargs = {
            "trust_remote_code": True,
            "use_cache": False if self.config["training"]["gradient_checkpointing"] else True,
        }
        
        # Add quantization for memory efficiency
        if self.config["gpu"]["optimize_memory"]:
            model_kwargs.update({
                "load_in_8bit": True,
                "device_map": "auto",
                "torch_dtype": torch.float16,
            })
        
        # Load model
        try:
            if self.config["model"]["model_path"]:
                model_name = self.config["model"]["model_path"]
            else:
                model_name = self.config["model"]["base_model"]
                
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Attempting to download model...")
            # Force download
            model = AutoModelForCausalLM.from_pretrained(
                self.config["model"]["base_model"],
                cache_dir=self.config["paths"]["cache_dir"],
                force_download=True,
                **model_kwargs
            )
        
        # Setup LoRA if enabled
        if self.config["model"]["use_lora"]:
            logger.info("Setting up LoRA...")
            model = prepare_model_for_kbit_training(model)
            
            lora_config = LoraConfig(
                r=self.config["model"]["lora_config"]["r"],
                lora_alpha=self.config["model"]["lora_config"]["lora_alpha"],
                target_modules=self.config["model"]["lora_config"]["target_modules"],
                lora_dropout=self.config["model"]["lora_config"]["lora_dropout"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        # Enable gradient checkpointing
        if self.config["training"]["gradient_checkpointing"]:
            model.gradient_checkpointing_enable()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
    
    def load_dataset(self, tokenizer):
        """Load and prepare dataset"""
        from datasets import load_dataset, Dataset
        import pandas as pd
        
        logger.info("Loading dataset...")
        
        # Try to load from different sources
        data_path = Path(self.config["data"]["data_path"])
        
        if (data_path / "train.json").exists():
            # Load from local JSON
            train_data = pd.read_json(data_path / "train.json")
            dataset = Dataset.from_pandas(train_data)
        elif (data_path / "train.csv").exists():
            # Load from local CSV
            train_data = pd.read_csv(data_path / "train.csv")
            dataset = Dataset.from_pandas(train_data)
        else:
            # Try to load from HuggingFace
            try:
                dataset = load_dataset(
                    f"FinGPT/{self.config['data']['dataset']}",
                    split="train",
                    cache_dir=self.config["paths"]["cache_dir"]
                )
            except:
                logger.warning("Dataset not found, creating sample data...")
                # Create sample dataset for testing
                sample_data = self.create_sample_dataset()
                dataset = Dataset.from_dict(sample_data)
        
        # Limit samples if specified
        if self.config["data"]["max_samples"]:
            dataset = dataset.select(range(min(len(dataset), self.config["data"]["max_samples"])))
        
        # Tokenize dataset
        def tokenize_function(examples):
            # Combine instruction and response
            if "instruction" in examples and "response" in examples:
                texts = [f"{inst} {resp}" for inst, resp in zip(examples["instruction"], examples["response"])]
            elif "text" in examples:
                texts = examples["text"]
            else:
                texts = examples[list(examples.keys())[0]]
            
            return tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=self.config["training"]["max_length"],
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
        
        # Split dataset
        if "train" not in tokenized_dataset:
            split_dataset = tokenized_dataset.train_test_split(
                test_size=1 - self.config["data"]["train_split"]
            )
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]
        else:
            train_dataset = tokenized_dataset
            eval_dataset = None
            
        logger.info(f"Dataset loaded: {len(train_dataset)} training samples")
        
        return train_dataset, eval_dataset
    
    def create_sample_dataset(self) -> Dict:
        """Create sample dataset for testing"""
        samples = {
            "instruction": [
                "Analyze AAPL stock for next week",
                "Predict MSFT price movement",
                "Evaluate GOOGL fundamentals",
            ],
            "response": [
                "Based on technical analysis, AAPL shows bullish momentum...",
                "MSFT is likely to see moderate gains due to...",
                "GOOGL fundamentals remain strong with...",
            ]
        }
        return samples
    
    def setup_training_args(self):
        """Setup training arguments with GPU optimization"""
        from transformers import TrainingArguments
        
        training_args = TrainingArguments(
            output_dir=self.config["paths"]["output_dir"],
            num_train_epochs=self.config["training"]["num_epochs"],
            per_device_train_batch_size=self.config["training"]["batch_size"],
            per_device_eval_batch_size=self.config["training"]["batch_size"],
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
            warmup_ratio=self.config["training"]["warmup_ratio"],
            learning_rate=self.config["training"]["learning_rate"],
            fp16=self.config["training"]["fp16"] and self.device.type == "cuda",
            logging_dir=self.config["paths"]["logs_dir"],
            logging_steps=self.config["training"]["logging_steps"],
            save_steps=self.config["training"]["save_steps"],
            eval_steps=self.config["training"]["eval_steps"],
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            report_to=["tensorboard"],
            push_to_hub=False,
            gradient_checkpointing=self.config["training"]["gradient_checkpointing"],
            optim="adamw_torch",
            dataloader_num_workers=4,
            remove_unused_columns=False,
        )
        
        # Add DeepSpeed if configured
        if self.config["gpu"]["use_deepspeed"]:
            training_args.deepspeed = self.create_deepspeed_config()
            
        return training_args
    
    def create_deepspeed_config(self) -> Dict:
        """Create DeepSpeed configuration for multi-GPU training"""
        return {
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True
            },
            "gradient_accumulation_steps": self.config["training"]["gradient_accumulation_steps"],
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "wall_clock_breakdown": False
        }
    
    def train(self):
        """Main training loop"""
        from transformers import Trainer, DataCollatorForLanguageModeling
        
        logger.info("Starting training setup...")
        
        # Check dependencies
        self.check_dependencies()
        
        # Setup model and tokenizer
        model, tokenizer = self.setup_model_and_tokenizer()
        
        # Load dataset
        train_dataset, eval_dataset = self.load_dataset(tokenizer)
        
        # Setup training arguments
        training_args = self.setup_training_args()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        
        # Start training
        logger.info("Starting training...")
        logger.info(f"GPU Memory before training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        try:
            trainer.train()
            
            # Save final model
            logger.info("Saving final model...")
            trainer.save_model(self.config["paths"]["output_dir"] + "/final_model")
            tokenizer.save_pretrained(self.config["paths"]["output_dir"] + "/final_model")
            
            logger.info("Training completed successfully!")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("GPU out of memory! Suggestions:")
                logger.error("1. Reduce batch_size in config")
                logger.error("2. Enable gradient_checkpointing")
                logger.error("3. Use 8-bit quantization")
                logger.error("4. Reduce max_length")
                torch.cuda.empty_cache()
            raise e
    
    def monitor_gpu(self):
        """Monitor GPU usage during training"""
        if torch.cuda.is_available():
            logger.info("=" * 50)
            logger.info("GPU Monitoring:")
            logger.info(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            logger.info(f"Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            logger.info(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
            
            # GPU utilization
            import subprocess
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    gpu_util, mem_util = result.stdout.strip().split(", ")
                    logger.info(f"GPU Utilization: {gpu_util}%")
                    logger.info(f"Memory Utilization: {mem_util}%")
            except:
                pass
            logger.info("=" * 50)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="FinGPT GPU Training Script")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--batch-size", type=int, help="Training batch size")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--model", type=str, help="Base model name or path")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--monitor", action="store_true", help="Enable GPU monitoring")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = FinGPTTrainer(args.config)
    
    # Override config with command line arguments
    if args.batch_size:
        trainer.config["training"]["batch_size"] = args.batch_size
    if args.epochs:
        trainer.config["training"]["num_epochs"] = args.epochs
    if args.lr:
        trainer.config["training"]["learning_rate"] = args.lr
    if args.model:
        trainer.config["model"]["base_model"] = args.model
    if args.output:
        trainer.config["paths"]["output_dir"] = args.output
    
    # Monitor GPU if requested
    if args.monitor:
        trainer.monitor_gpu()
    
    # Start training
    trainer.train()
    
    # Final GPU stats
    trainer.monitor_gpu()


if __name__ == "__main__":
    main()