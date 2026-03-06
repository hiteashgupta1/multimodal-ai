import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from auto_retrain import get_next_model_version

BASE_MODEL = "distilgpt2"
DATA_PATH = "training/train.json"
version = get_next_model_version()
OUTPUT_DIR = f"models/orchestrator_v{version}"

print("Saving model version:", version)

# -----------------------------
# LOAD TOKENIZER
# -----------------------------

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# LOAD DATASET
# -----------------------------

dataset = load_dataset("json", data_files=DATA_PATH)["train"]

def preprocess(example):

    instruction = example.get("instruction") or example.get("input") or ""
    output = example.get("output") or ""

    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

    tokens = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=512
    )

    tokens["labels"] = tokens["input_ids"]

    return tokens

dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

dataset.set_format("torch")

# -----------------------------
# LOAD MODEL (CPU SAFE)
# -----------------------------

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model.to("cpu")

# -----------------------------
# LORA CONFIG
# -----------------------------

lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["c_attn"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# -----------------------------
# TRAINING ARGUMENTS
# -----------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    num_train_epochs=1,
    logging_steps=5,
    save_strategy="epoch",
    remove_unused_columns=False,
    dataloader_pin_memory=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# -----------------------------
# TRAIN
# -----------------------------

trainer.train()

# -----------------------------
# SAVE MODEL
# -----------------------------

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("LoRA training complete.")