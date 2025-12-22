from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.utils import dispatch_for_generation

# Select model and load it.
MODEL_ID = "../Qwen/Qwen3-0.6B"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 256 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)

print(ds)
print(ds[-1])

def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)
print(ds)
print(ds[-1])

# Tokenize inputs.
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


# Configure the quantization algorithm to run.
# NOTE: vllm currently does not support asym MoE, using symmetric here
recipe = [
    AWQModifier(
        ignore=["lm_head", "re:.*mlp.gate$", "re:.*mlp.shared_expert_gate$"],
        scheme="W4A16",
        targets=["Linear"],
    ),
]

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
inputs = tokenizer("Hello my name is", return_tensors="pt").to(model.device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

output = model.generate(
    **inputs, 
    max_new_tokens=100,
    do_sample=True,            # 开启采样，减少死循环
    top_p=0.9,                 # 增加多样性
    temperature=0.7,
    repetition_penalty=1.1,    # 显式惩罚重复
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)

print(tokenizer.decode(output[0]), skip_special_tokens=True)
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-awq-sym"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)