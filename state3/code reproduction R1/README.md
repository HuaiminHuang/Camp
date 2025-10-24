# R1: SFT + GRPO Training Setup with vLLM
(原帖)[https://github.com/QunBB/DeepLearning/tree/main/llms/train]

下载速度慢的可以切换到清华镜像
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
```

## Environment Setup

Make sure you have Python >= 3.10 and CUDA installed. Recommended packages:

```bash
conda create -n R1Think python=3.10
conda activate R1Think


pip install \
    torch==2.8.0 \
    accelerate==1.10.1 \
    transformers==4.57.1 \
    datasets==4.2.0 \
    vllm==0.11.0 \
    trl==0.18.2
```

# 一、SFT

```bash
python main.py --task=sft_train \
--model_name_or_path=Qwen/Qwen2.5-0.5B-Instruct \
--bf16 \
--checkpoint_dir=outputs/Qwen-0.5B-SFT-FirstHalf \
--per_device_train_batch_size=8 \
--save_strategy=epoch \
--epochs=1
```

# 二、GRPO



## 2️⃣ Distributed Training Environment Variables

If using single-machine or multi-GPU training, set these before running main.py:


```py
import os

os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

```

MASTER_ADDR / MASTER_PORT → master node for distributed training  
RANK / LOCAL_RANK → process rank  
WORLD_SIZE → total number of processes

3️⃣ vLLM Acceleration

vLLM can significantly speed up generation in GRPO/PPO training.

- GRPOConfig vLLM Options

```py
training_args = GRPOConfig(
    ...
    use_vllm=True,                       # Enable vLLM acceleration
    vllm_mode="colocate",                # "colocate" or "server"
    vllm_gpu_memory_utilization=0.3,     # Fraction of GPU memory vLLM can use
    enforce_eager=False                   # Fix Pydantic validation error
)

```

- vllm_mode

  - "server": Run vLLM in a separate server process (trl vllm-serve)

  - "colocate": Run vLLM in the same process as training

- vllm_gpu_memory_utilization: Prevents OOM by limiting vLLM GPU usage

- enforce_eager=False: Fixes Pydantic Core type validation errors


```bash
Example: Running a vLLM server (optional if colocate)
trl vllm-serve --model outputs/Qwen-0.5B-SFT-FirstHalf/checkpoint-234 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.4 \
  --enforce-eager False
```

Args:

  1. `max-model-len`: Maximum input + output sequence length

  2. `gpu-memory-utilization`: Fraction of GPU memory for vLLM

  3. `tensor-parallel-size`: Tensor parallelism size (1 if single GPU)

  4. `enforce-eager`: Same as above


# 4️⃣ GRPO Training Command
```bash
python main.py \
  --task=grpo_train \
  --model_name_or_path=outputs/Qwen-0.5B-SFT-FirstHalf/checkpoint-234 \
  --checkpoint_dir=outputs/Qwen-0.5B-GRPO-SecondHalf \
  --bf16 \
  --use_vllm \
  --split_half=second_half \
  --per_device_train_batch_size=4 \
  ----gradient_accumulation_steps=4 \
  --save_strategy=epoch
```

# 三、Inference 


```bash
python main.py --task=inference --checkpoint_dir=outputs/Qwen-0.5B-GRPO-SecondHalf/checkpoint-934
```

```
请输入你的问题：
Natalia sold clips to 22 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

Assistant:
<think>
In April, Natalia sold clips to 22 friends.
In May, she sold half as many clips as in April, which is 22/2 = <<22/2=11>>11 clips.
  Altogether, Natalia sold 22+11 = <<22+11=33>>33 clips in April and May.
</think>
<answer>
33
</answer>
```
