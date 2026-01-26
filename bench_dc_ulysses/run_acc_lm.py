import os
import argparse
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
from deepspeed.utils.timer import SynchronizedWallClockTimer

import torch
import random
import numpy as np
import csv
import time
import os

from distributed_attention import ulysses_attention_forward
from ring_attention import ring_attention_forward
from sp_dp_registry import get_group, register_groups

torch.set_float32_matmul_precision("high")

## This dictionary is globally should be globally visible everywhere. ##
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = 12 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--dataset_name", type=str, default="timdettmers/openassistant-guanaco")
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--compile", type=str, default="deepcompile")
    parser.add_argument("--passes", type=str, default=None)
    parser.add_argument("--backend", type=str, default="inductor")
    parser.add_argument("--offload_opt_states", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile_memory", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--profile_dir", type=str, default="profiles")
    parser.add_argument("--bench_step", type=int, default=1)
    parser.add_argument("--warmup_step", type=int, default=15)
    parser.add_argument("--print_interval", type=int, default=1)
    parser.add_argument("--experiment_folder", type=str, default="")
    parser.add_argument("--sp_size", type=int, default=2)
    parser.add_argument("--dp_size", type=int, default=1)


    return parser.parse_args()

def main():
    args = get_args()
    set_seed(12)

    if args.deterministic:
        enable_full_determinism(12)
        from torch._inductor import config
        config.fallback_random = True
        torch.use_deterministic_algorithms(True)

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    device = accelerator.device
    is_deepspeed = accelerator.state.deepspeed_plugin is not None
    assert accelerator.num_processes == args.sp_size * args.dp_size, 'Incorrect dp/sp sizing'

    # Load model and tokenizer
    if accelerator.is_main_process:
        print("Loading model and tokenizer...")

    model_name = args.model_name
    if args.compile == "deepcompile":
        attention_backend = "sdpa"
    else:
        if args.compile == "eager" or args.compile == "compile":
            from transformers.models.llama import modeling_llama
            attention_backend = "ulyssess"
            modeling_llama.ALL_ATTENTION_FUNCTIONS["ulyssess"] = ulysses_attention_forward
        elif args.compile == "ringattn":
            from transformers.models.llama import modeling_llama
            attention_backend = "ringattn"
            modeling_llama.ALL_ATTENTION_FUNCTIONS["ringattn"] = ring_attention_forward 

    if args.num_layers is not None:
        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        print(f"num_hidden_layers: {model_config.num_hidden_layers} -> {args.num_layers}")
        model_config.num_hidden_layers = args.num_layers
        model_config._attn_implementation = attention_backend
        model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)
    else:
        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_config._attn_implementation = attention_backend
        model = AutoModelForCausalLM.from_pretrained(model_name, config=model_config, trust_remote_code=True)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    ## Instantiate process groups for SP+DP interoperation. ##
    os.environ['SP_SIZE'] = str(args.sp_size)
    os.environ['DP_SIZE'] = str(args.dp_size)
    
    if args.compile in ["eager", "compile", "ringattn"]:
        ## We register in the run_acc_lm.py file for baselines to reduce code-duplication.
        ## Else the registration happens within the SP compiler pass within deepspeed.
        group_listing = []
        offset = 0
        for _ in range(args.dp_size):
            group_listing.append([i + offset for i in range(args.sp_size)])
            offset += args.sp_size

        register_groups(group_listing)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    if accelerator.is_main_process:
        print("Loading dataset...")

    g = torch.Generator()
    g.manual_seed(12)
    dataset = load_dataset('ag_news', split='train[:1%]')

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(2)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', max_length=10, truncation=True) ## Fix max_length and generate fake data instead to not exhaust disk.

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    num_replicas_ = args.dp_size
    rank_ = accelerator.process_index // args.sp_size

    sampler = DistributedSampler(tokenized_dataset, num_replicas=num_replicas_, rank=rank_, seed=12, shuffle=False)  
    data_loader = DataLoader(tokenized_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, worker_init_fn=seed_worker, generator=g)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)
    print(f"Model prepared: {model.__class__}")

    ### CUSTOM DEBUGGING ###
    ## everything tried induces a graph break. ## 
    ### CUSTOM DEBUGGING -- END ###

    if args.compile == "deepcompile":
        print(f"Running deepcompile with backend={args.backend}")
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._dynamo.config.capture_scalar_outputs = True
        model.compile(backend=args.backend)
    elif args.compile == "compile" or args.compile == "ringattn":
        print(f"Running torch.compile with backend={args.backend}")
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._dynamo.config.capture_scalar_outputs = True
        model = torch.compile(model, backend=args.backend)
    else:
        print(f"Running eager")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model_name = args.model_name.split("/")[-1]
    exp_name = f"{model_name}_np{accelerator.num_processes}_{args.compile}_" \
               f"B{args.backend}_" \
               f"L{0 if args.num_layers is None else args.num_layers}_" \
               f"bs{args.batch_size}_seq{args.seq_length}_" \
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
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) if do_profile else nullcontext()

    if args.profile_memory and accelerator.is_main_process:
        torch.cuda.memory._record_memory_history(
            max_entries=100000
        )
    
    # Training loop
    model.train()
    global_step = 0

    iter_times = []
    memory = []

    print(f"Using global sequence length: {args.seq_length}")

    # See https://github.com/microsoft/DeepSpeed/issues/6793
    acc_context = nullcontext if is_deepspeed else accelerator.accumulate

    os.makedirs("logs", exist_ok=True)
    loss_log_file = open(f"logs/loss_{args.compile}_{args.seq_length}_{accelerator.process_index}.csv", "w")
    loss_log_file.write("step,loss\n")
    ## We register some hooks to later print activations.
    activations = {}

    ## Set profiling flag accordingly. ##
    profile = False

    with prof_context as prof:
        for epoch in range(args.num_epochs):
            start_iter = time.time()

            for step, batch in enumerate(data_loader):
                input_ids = torch.tensor([
                    [1 for _ in range(args.seq_length)]
                ], device=device)
                attention_mask = torch.tensor([
                    [1] * args.seq_length
                ], device=device)
                start = accelerator.process_index * args.seq_length // accelerator.num_processes
                end = start + args.seq_length // accelerator.num_processes
                
                if args.compile in ("compile", "eager", "ringattn"):
                    start = accelerator.process_index * args.seq_length // accelerator.num_processes
                    end = start + args.seq_length // accelerator.num_processes
                    input_ids = input_ids[:, start:end]
                    attention_mask = attention_mask[:, start:end]
                    position_ids = torch.arange(start, end, device=device).unsqueeze(0)
                else:
                    start = accelerator.process_index * args.seq_length // accelerator.num_processes
                    end = start + args.seq_length // accelerator.num_processes
                    input_ids = input_ids[:, start:end]
                    position_ids = None
                
                with acc_context(model):
                    torch.cuda.reset_peak_memory_stats(device)
                    for _ in range(1):
                        if profile:
                            torch.cuda.synchronize()
                            fwd_usage_prior = SynchronizedWallClockTimer.memory_usage()
                            fwd_start = time.time()
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, position_ids=position_ids)
                        if profile:
                            torch.cuda.synchronize()
                            fwd_end = time.time()
                            fwd_usage_post = SynchronizedWallClockTimer.memory_usage()
                            print('fwd time: ' + str(fwd_end-fwd_start))
                            print('fwd memory prior: ' + str(fwd_usage_prior))
                            print('fwd memory post: ' + str(fwd_usage_post))
                        loss = outputs.loss
                        #print(f'outputs: {outputs} [rank: {dist.get_rank()}]')

                        update_step = (is_deepspeed and model.is_gradient_accumulation_boundary()) \
                            or (not is_deepspeed and accelerator.sync_gradients)
                        
                        loss_log_file.write(f"{global_step},{loss.item()}\n")
                        loss_log_file.flush()

                        accelerator.backward(loss)

                        optimizer.step()
                        optimizer.zero_grad()

                    ## PERF benchmarking code, ensure warmup prior. ##
                    #torch.cuda.synchronize()
                    #start = time.time()
                    #for _ in range(100):
                    #    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, position_ids=position_ids)
                    #    loss = outputs.loss
                    #    #print(f'outputs: {outputs} [rank: {dist.get_rank()}]')

                    #    update_step = (is_deepspeed and model.is_gradient_accumulation_boundary()) \
                    #        or (not is_deepspeed and accelerator.sync_gradients)
                    #    
                    #    #loss_log_file.write(f"{global_step},{loss.item()}\n")
                    #    #loss_log_file.flush()

                    #    accelerator.backward(loss)

                    #    optimizer.step()
                    #    optimizer.zero_grad()
                    #torch.cuda.synchronize()
                    #end = time.time()
                    #print(f'time taken: {end-start}')

                    global_step += 1

                    if update_step:
                        print(f"Epoch {epoch+1}, Step {global_step}, Loss: {loss.item()} time: {time.time() - start_iter} alloc_mem: {torch.cuda.memory_allocated() / (1024 ** 3)} peak_mem: {torch.cuda.max_memory_allocated() / (1024 ** 3)}")

                        iter_times.append(time.time() - start_iter)
                        memory.append(torch.cuda.max_memory_allocated() / (1024 ** 3))
                        start_iter = time.time()

                if do_profile:
                    prof.step()

                if global_step >= args.bench_step * args.gradient_accumulation_steps:
                    break

    torch.cuda.synchronize()

    if args.profile_memory and accelerator.is_main_process:
        torch.cuda.memory._dump_snapshot(f"{args.compile}_{args.seq_length}_ulysses.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

    iter_times = iter_times[args.warmup_step:]
    loss_log_file.close()

    if accelerator.is_main_process:
        compile_time_sum = 0
        compile_time = 0
        if args.compile and hasattr(model, "get_compile_time"):
            compile_time = model.get_compile_time()
            compile_time_sum = sum(t for _, _, _, t in compile_time)

        is_deepcompile = is_deepspeed and model._config.compile_config.deepcompile
        msg = f"{args.model_name} ds={is_deepspeed} np={accelerator.num_processes} batch_size={args.batch_size} seq={args.seq_length} compile={args.compile} backend={args.backend} deepcompile={is_deepcompile} compile_time={compile_time_sum} iteration time: {sum(iter_times) / (1 + len(iter_times)):.4f} alloc_mem: {torch.cuda.memory_allocated()} peak_mem: {torch.cuda.max_memory_allocated()}"
        print(msg)

        if args.profile_dir:
            from pathlib import Path
            filepath = Path(args.profile_dir) / f"result.txt"
            with open(filepath, "a") as f:
                f.write(f"{timestamp} {msg}" + "\n")

            if args.compile != "eager":
                filepath = Path(args.profile_dir) / f"compile_time.txt"
                with open(filepath, "a") as f:
                    msg =  f"{msg} compile_time={compile_time_sum} {compile_time}"
                    f.write(f"{timestamp} {msg}" + "\n")
        
        if args.profile and args.experiment_folder != "":
            avg_iter = sum(iter_times) / len(iter_times)
            avg_memory = sum(memory) / len(memory)

            output_folder = os.path.join("experiments", args.experiment_folder)
            os.makedirs(output_folder, exist_ok=True)

            csv_name = f"{model_name}_c{args.compile}_bs{args.batch_size}_seq{args.seq_length}_T{timestamp}.csv"
            csv_file = os.path.join(output_folder, csv_name)

            with open(csv_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["batch", "seq_len", "compile", "time_per_iter", "peak_memory"])
                writer.writerow([args.batch_size, args.seq_length, args.compile, avg_iter, avg_memory])

if __name__ == "__main__":
    torch._dynamo.config.accumulated_cache_size_limit = 256
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False
    try:
        main()
    except Exception as e:
        import pdb
        import traceback
        traceback.print_exc()
        pdb.post_mortem()
    finally:
        # Ensure distributed resources are cleaned up to avoid leaks/warnings
        if dist.is_available() and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            try:
                dist.destroy_process_group()
            except Exception:
                pass
