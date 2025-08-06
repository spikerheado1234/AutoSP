#!/bin/bash
#SBATCH --account=bcjw-delta-gpu 
#SBATCH --nodes=2
#SBATCH --partition=gpuA100x4
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

conda activate dc

export NCCL_DEBUG=INFO
export LOGLEVEL=INFO
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip


PROFILE_DIR=${PROFILE_DIR:-$HOME/desktop/bench/profiles}
mkdir -p ${PROFILE_DIR}

BATCH_SIZE=1
SEQ_LENGTH=6144
BACKEND="deepspeed"
NUM_NODES=2
NGPUS_PER_NODE=1
NUM_PROCESSES=2
GRADIENT_ACCUMULATION_STEPS=1
ACTIVATION_CHECKPOINTING=0
NUM_LAYERS=2
COMPILE=0
DEEPCOMPILE=0
EXTRA_OPTS="--num_layers ${NUM_LAYERS} --seq_length ${SEQ_LENGTH} --batch_size ${BATCH_SIZE} --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} --profile --profile_dir ${PROFILE_DIR}"
LOG_FILE=debug_b${BACKEND}np${NUM_PROCESSES}c${COMPILE}dc${DEEPCOMPILE}bs${BATCH_SIZE}seq${SEQ_LENGTH}.log

mkdir -p logs

CONFIG_YAML_TEMPLATE=$HOME/desktop/bench/configs/ds_config.yaml.template
CONFIG_JSON_TEMPLATE=$HOME/desktop/bench/configs/ds_config.json.template

python generate_conf.py \
    --num_machines ${NUM_NODES} \
    --num_processes ${NUM_PROCESSES} \
    --template_file ${CONFIG_YAML_TEMPLATE} \
    --output_file configs/config.yaml

python generate_conf.py \
        --num_machines ${NUM_NODES} \
        --num_processes ${NUM_PROCESSES} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
        --template_file ${CONFIG_JSON_TEMPLATE} \
        --output_file configs/ds_config.json

srun torchrun \
--nnodes 2 \
--nproc_per_node 1 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint ${head_node_ip}:29511 \
$HOME/desktop/bench/my_ulysses.py \
${EXTRA_OPTS} \
2>&1 | tee ${LOG_FILE}