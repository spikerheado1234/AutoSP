import os
import argparse
import time
from datetime import datetime
from typing import List

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, enable_full_determinism
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist
import random
import numpy as np
from torch import profiler

import deepspeed
import deepspeed.comm as dist

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--dataset_name", type=str, default="timdettmers/openassistant-guanaco")
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--passes", type=str, default=None)
    parser.add_argument("--backend", type=str, default="inductor")
    parser.add_argument("--offload_opt_states", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--profile_dir", type=str, default=None)
    parser.add_argument("--bench_step", type=int, default=30)
    parser.add_argument("--warmup_step", type=int, default=15)
    parser.add_argument("--print_interval", type=int, default=10)
    return parser.parse_args()

def main():
    args = get_args()
    set_seed(12)

    # Initialize distributed training via DeepSpeed
    deepspeed.init_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"World size: {world_size}, local_rank: {local_rank}")
    
    torch.cuda.set_device(local_rank)
    
    # Load and customize model config
    model_name = args.model_name
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    print(f"num_hidden_layers: {model_config.num_hidden_layers} -> {args.num_layers}")
    model_config.num_hidden_layers = args.num_layers
    model_config.update({"seq_length": args.seq_length})
    
    # Create model from config (no pretrained weights)
    model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)
    print(model)

    # Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset (just 1% of train split)
    print("Loading dataset...")
    dataset = load_dataset('ag_news', split='train[:2%]')

    # Re-initialize tokenizer to set special pad token
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(2)

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            padding='max_length', 
            max_length=args.seq_length, 
            truncation=True
        )

    # Map over dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    # Create sampler and dataloader
    data_loader = DataLoader(tokenized_dataset, 
                             batch_size=args.batch_size,  
                             num_workers=4)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Move to device and wrap in DDP
    model = model.cuda()

    model.train()
    global_step = 0

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(data_loader):
            
            input_ids = batch['input_ids'].to(local_rank)
            attention_mask = batch['attention_mask'].to(local_rank)
            
            # Forward pass
            if global_step == 30:
                torch.cuda.memory._record_memory_history()
            if global_step == 30:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    profile_memory=True,  
                    record_shapes=True,   
                    with_stack=True       
                ) as prof:
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_max_memory_allocated()
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                    loss = outputs.loss
                    avg_loss = loss.clone()
                    torch.distributed.all_reduce(avg_loss, op=torch.distributed.ReduceOp.AVG)
                    avg_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    prof.step()
                prof.export_chrome_trace("memory_trace.json.gz")
            else:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_max_memory_allocated()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                avg_loss = loss.clone()
                torch.distributed.all_reduce(avg_loss, op=torch.distributed.ReduceOp.AVG)
                avg_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            if global_step == 30:
                torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
            
            if global_step == 30:
                alloc_mem_mb = torch.cuda.memory_allocated(local_rank) / (1024 ** 2)
                peak_alloc_mem_mb = torch.cuda.max_memory_allocated(local_rank) / (1024 ** 2)
                cached_mem_mb = torch.cuda.memory_reserved(local_rank) / (1024 ** 2)
                peak_cached_mem_mb = torch.cuda.max_memory_reserved(local_rank) / (1024 ** 2)

                profiling_str = (
                    f"[Rank {local_rank}] Step {global_step} (Profiling Training),\n"
                    f"  Alloc_Mem:       {alloc_mem_mb:.2f} MB\n"
                    f"  Peak_Alloc_Mem:  {peak_alloc_mem_mb:.2f} MB\n"
                    f"  Cached_Mem:      {cached_mem_mb:.2f} MB\n"
                    f"  Peak_Cached_Mem: {peak_cached_mem_mb:.2f} MB\n"
                )
                profiling_str += f"\n***** Final Memory Summary (Rank {local_rank}) *****\n"
                profiling_str += torch.cuda.memory_summary(abbreviated=True)

                if args.profile_dir:
                    os.makedirs(args.profile_dir, exist_ok=True)
                    profile_path = os.path.join(args.profile_dir, "memory.txt")
                    with open(profile_path, "a") as f:
                        f.write(profiling_str + "\n\n")
                    snapshot_path = os.path.join(args.profile_dir, f"snapshot_rank{local_rank}.pickle")
                    
            global_step += 1
            
            if global_step == 31:
                break
            
if __name__ == "__main__":
    main()
