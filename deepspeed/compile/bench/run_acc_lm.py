import os
import argparse
import time
from datetime import datetime
from contextlib import nullcontext
from typing import List

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, enable_full_determinism
from datasets import load_dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.nn.functional as F


from patch_phi3_moe import patch_phi3moe

import torch
import random
import numpy as np

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
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--dataset_name", type=str, default="timdettmers/openassistant-guanaco")
    parser.add_argument("--num_layers", type=int, default=0)
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


def make_schedule(passes: List[str], warmup):
    schedule = [(0, ["zero3_compile"])]
    schedule.append((warmup, ["zero3_compile"] + passes))
    return schedule


def main():
    args = get_args()
    set_seed(12)
    if args.deterministic:
        enable_full_determinism(0)
        from torch._inductor import config
        config.fallback_random = True

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    device = accelerator.device
    is_deepspeed = accelerator.state.deepspeed_plugin is not None
    print(f"Running on device: {device} is_deepspeed: {is_deepspeed}")

    # Load model and tokenizer
    if accelerator.is_main_process:
        print("Loading model and tokenizer...")

    model_name = args.model_name

    if args.num_layers > 0:
        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        print(f"num_hidden_layers: {model_config.num_hidden_layers} -> {args.num_layers}")
        model_config.num_hidden_layers = args.num_layers
        # update attention backend
        model_config._attn_implementation = "sdpa"
        model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True, attn_implementation="sdpa")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        

    if patch_phi3moe(model) and accelerator.is_main_process:
        print("Patched Phi-3.5-MoE model")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    if accelerator.is_main_process:
        print("Loading dataset...")
        
    dataset = load_dataset('ag_news', split='train[:1%]')

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(2)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', max_length=args.seq_length, truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    sampler = DistributedSampler(tokenized_dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index)
    data_loader = DataLoader(tokenized_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)

    # Prepare optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Prepare everything with accelerator
    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)
    print(model)
    print(f"Model prepared: {model.__class__}")

    if is_deepspeed:
        if args.compile:
            torch._dynamo.config.capture_dynamic_output_shape_ops = True
            torch._dynamo.config.capture_scalar_outputs = True

            schedule = make_schedule(args.passes.split(","), warmup=5) if args.passes else None
            model.compile(backend=args.backend, schedule=schedule)
    else:
        if args.compile:
            model = torch.compile(model, backend=args.backend)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model_name = args.model_name.split("/")[-1]
    exp_name = f"{model_name}_np{accelerator.num_processes}ds{1 if is_deepspeed else 0}" \
               f"B{args.backend}" \
               f"L{0 if args.num_layers is None else args.num_layers}" \
               f"bs{args.batch_size}seq{args.seq_length}acc{args.gradient_accumulation_steps}ac{1 if args.activation_checkpointing else 0}" \
               f"pass_{'none' if args.passes is None else args.passes.replace(',', '_')}_" \
               f"os{1 if args.offload_opt_states else 0}" \
               f"T{timestamp}"
    if args.profile_dir:
        if accelerator.is_main_process and args.profile_dir:
            os.makedirs(args.profile_dir, exist_ok=True)
            if args.profile:
                prof_dir = f"{args.profile_dir}/{exp_name}"
                os.makedirs(prof_dir, exist_ok=True)
        accelerator.wait_for_everyone()        
        
    do_profile = args.profile and accelerator.is_main_process
    prof_context = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=10*args.gradient_accumulation_steps, active=4, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(prof_dir),
    ) if do_profile else nullcontext()

    # Training loop
    model.train()
    global_step = 0

    iter_times = []

    # See https://github.com/microsoft/DeepSpeed/issues/6793
    acc_context = nullcontext if is_deepspeed else accelerator.accumulate

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    with prof_context as prof:
        for epoch in range(args.num_epochs):
            start_iter = time.time()

            for step, batch in enumerate(data_loader):
                input_ids = batch['input_ids'].to(device)             # [B, S]
                attention_mask = batch['attention_mask'].to(device)   # [B, S]
                B, S = input_ids.shape

                # sequence shard
                chunk_size = S // world_size
                start = rank * chunk_size
                end = (rank + 1) * chunk_size
                input_ids_shard = input_ids[:, start:end]               # [B, S_shard]
                labels_shard = input_ids[:, start:end]                  # [B, S_shard]

                with acc_context(model):
                    torch.cuda.reset_peak_memory_stats()
                    outputs = model(
                        input_ids=input_ids_shard,
                        attention_mask=attention_mask,
                        use_cache=False,
                        labels=labels_shard  
                    )
                    logits = outputs.logits  # [B, S_shard, V]

                    # Local loss computation
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),     # [B * S_shard, V]
                        labels_shard.view(-1),                # [B * S_shard]
                        ignore_index=-100,
                        reduction='mean'
                    )
                    
                    update_step = (is_deepspeed and model.is_gradient_accumulation_boundary()) \
                        or (not is_deepspeed and accelerator.sync_gradients)
                    accelerator.backward(loss)

                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    if update_step:
                        if accelerator.is_main_process and global_step % (args.print_interval * args.gradient_accumulation_steps) == 0:
                            print(f"Epoch {epoch+1}, Step {global_step}, Loss: {loss.item()} sync: {accelerator.sync_gradients} time: {time.time() - start_iter} alloc_mem: {torch.cuda.memory_allocated()} peak_mem: {torch.cuda.max_memory_allocated()}")

                        iter_times.append(time.time() - start_iter)
                        start_iter = time.time()

                if do_profile:
                    prof.step()

                if global_step > args.bench_step * args.gradient_accumulation_steps:
                    break

    iter_times = iter_times[args.warmup_step:]

    if accelerator.is_main_process:
        compile_time_sum = 0
        compile_time = 0
        if args.compile and hasattr(model, "get_compile_time"):
            compile_time = model.get_compile_time()
            compile_time_sum = sum(t for _, _, _, t in compile_time)

        is_deepcompile = is_deepspeed and model._config.compile_config.deepcompile
        msg = f"{args.model_name} ds={is_deepspeed} np={accelerator.num_processes} batch_size={args.batch_size} seq={args.seq_length} acc={args.gradient_accumulation_steps} ac={args.activation_checkpointing} compile={args.compile} backend={args.backend} deepcompile={is_deepcompile} passes={args.passes} compile_time={compile_time_sum} iteration time: {sum(iter_times) / (1 + len(iter_times)):.4f} alloc_mem: {torch.cuda.memory_allocated()} peak_mem: {torch.cuda.max_memory_allocated()}"
        print(msg)

        if args.profile_dir:
            from pathlib import Path
            filepath = Path(args.profile_dir) / f"result.txt"
            with open(filepath, "a") as f:
                f.write(f"{timestamp} {msg}" + "\n")

            if args.compile:
                filepath = Path(args.profile_dir) / f"compile_time.txt"
                with open(filepath, "a") as f:
                    msg =  f"{msg} compile_time={compile_time_sum} {compile_time}"
                    f.write(f"{timestamp} {msg}" + "\n")

    # # Save the model
    # if accelerator.is_main_process:
    #     accelerator.wait_for_everyone()
    #     unwrapped_model = accelerator.unwrap_model(model)
    #     unwrapped_model.save_pretrained("fine_tuned_model", save_function=accelerator.save)
    #     tokenizer.save_pretrained("fine_tuned_model")

if __name__ == "__main__":
    main()