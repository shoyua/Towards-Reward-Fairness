########################
# This script is modified from the TRL package https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/reward_modeling.py
# This script is designed for the reward modeling with Mistral model which should be handled carefully because it does not have an official pad token
# If you have any question, feel free to send me an email via wx13@illinois.edu
########################
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
import json

# from peft import LoraConfig, TaskTypes, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy

import torch.nn.functional as F
from torch.utils.data import Dataset



class CustomDataset(Dataset):
    def __init__(self, file_path, tokenize):
        self.tokenize = tokenize
        self.data = []
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():  
                    self.data.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        chosen_text = item["chosen"]
        rejected_text = item["rejected"]
        chosen_score = torch.tensor(item["chosen_score"], dtype=torch.float32)
        rejected_score = torch.tensor(item["rejected_score"], dtype=torch.float32)
        sample = {
            "chosen": chosen_text,
            "rejected": rejected_text,
            "chosen_score": chosen_score,
            "rejected_score": rejected_score,
        }
        return self.tokenize(sample)


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"}
    )

    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    # for 8 GPU, the global batch size is 512
    gradient_accumulation_steps: Optional[int] = field(default=64)
    learning_rate: Optional[float] = field(default=2e-6)
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    moe: Optional[str] = (
        field(
            default="moe",
            metadata={
                "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
            },
        ),
    )
    mode: Optional[str] = (
        field(
            default="bt",
            metadata={
                "help": "`bt`, `fr`, and `fc`, which stand for Bradley-Terry (BT) model, Fairness Regularization, and Fairness Coefficient, respectively."
            },
        ),
    )
    alpha: Optional[float] = field(default=0.1)
    beta: Optional[float] = field(default=-1)
    gamma: Optional[float] = field(default=0.5)
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    train_set_path: Optional[str] = field(
        default="hendrydong/preference_700K",
        metadata={"help": "The dir of the subset of the training data to use"},
    )
    eval_set_path: Optional[str] = field(
        default="hendrydong/preference_700K",
        metadata={"help": "The dir of the subset of the eval data to use"},
    )
    output_path: Optional[str] = field(
        default="./models/llama3_rm",
        metadata={"help": "The dir for output model"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        # default="adamw_hf",
        default="paged_adamw_32bit",
        # default="adamw_torch_fused",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=4096)

    save_every_steps: Optional[int] = field(
        default=500,
        metadata={"help": "Save the model every x steps"},
    )
    eval_every_steps: Optional[int] = field(
        default=999999,
        metadata={"help": "Eval the model every x steps"},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Load the value-head model and tokenizer.
tokenizer_name = script_args.model_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)

# Adjusted according to the base model
# Need to do this for the models that don't have an official pad token.
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
print(tokenizer.padding_side)
tokenizer.truncation_side = "left"
tokenizer.model_max_length = script_args.max_length
# tokenizer.padding_side = "right"


# Get the dataset
train_path = script_args.train_set_path
eval_path = script_args.eval_set_path
output_name = script_args.output_path


def build_dataset(tokenizer, train_path, eval_path):

    def tokenize(sample):
        if tokenizer.bos_token:
            sample["positive"] = tokenizer.apply_chat_template(
                sample["chosen"], tokenize=False, add_generation_prompt=False
            ).replace(tokenizer.bos_token, "")
            sample["negative"] = tokenizer.apply_chat_template(
                sample["rejected"], tokenize=False, add_generation_prompt=False
            ).replace(tokenizer.bos_token, "")
        else:
            sample["positive"] = tokenizer.apply_chat_template(
                sample["chosen"], tokenize=False, add_generation_prompt=False
            )
            sample["negative"] = tokenizer.apply_chat_template(
                sample["rejected"], tokenize=False, add_generation_prompt=False
            )
        tokenized_pos = tokenizer(sample["positive"], truncation=True)
        tokenized_neg = tokenizer(sample["negative"], truncation=True)
        sample["input_ids_j"] = tokenized_pos["input_ids"]  
        sample["attention_mask_j"] = tokenized_pos["attention_mask"]
        sample["input_ids_k"] = tokenized_neg["input_ids"]
        sample["attention_mask_k"] = tokenized_neg["attention_mask"]
        return sample

    if "HH-RLHF-Helpful-standard" in train_path:
        ds = load_dataset(train_path, split="train").shuffle(seed=42)
        ds = ds.map(tokenize, num_proc=8)
    else:
        ds = CustomDataset(train_path, tokenize)
    # ds = ds.select(range(2000))

    eval_dataset = None

    train_dataset = ds
    # eval_dataset = load_dataset(eval_path, split="train").shuffle(seed=42).select(range(500))
    eval_dataset = ds  # .select(range(500))
    return train_dataset, eval_dataset


train_dataset, eval_dataset = build_dataset(tokenizer, train_path, eval_path)
print("Training set: ", len(train_dataset), " Eval set: ", len(eval_dataset))

# Define the trainer


import wandb

# wandb.init(project="fair-RM", name=script_args.output_path)


# Define the trainer
training_args = TrainingArguments(
    output_dir=output_name,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    evaluation_strategy="steps",
    eval_steps=script_args.eval_every_steps,
    save_strategy="steps",
    save_steps=script_args.save_every_steps,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    remove_unused_columns=False,
    label_names=[],
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=10,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_ratio=0.03,
    report_to="wandb",
    
)

model = AutoModelForSequenceClassification.from_pretrained(
    script_args.model_name,
    num_labels=1,
    torch_dtype=torch.bfloat16,
    use_flash_attention_2=True,
)
if script_args.moe == "moe":
    in_features = model.score.in_features

    
    class MoELayer(nn.Module):
        def __init__(self, in_features):
            super(MoELayer, self).__init__()
            self.v_head1 = nn.Linear(in_features, 1)
            self.v_head2 = nn.Linear(in_features, 1)
            self.v_head3 = nn.Linear(in_features, 1)
            self.v_head4 = nn.Linear(in_features, 1)
            self.v_head5 = nn.Linear(in_features, 1)
            self.v_head6 = nn.Linear(in_features, 1)
            num_head = 6
            self.gate_layer = nn.Linear(in_features, num_head, bias=False)

        def forward(self, x):
            # print("<>"*10)
            # print(x.shape) #([1, 685, 4096])
            gate_values = F.softmax(self.gate_layer(x), dim=2)  # [1,685,4]
            expert1 = self.v_head1(x)  # [1,685,1]
            expert2 = self.v_head2(x)
            expert3 = self.v_head3(x)
            expert4 = self.v_head4(x)
            expert5 = self.v_head5(x)
            expert6 = self.v_head6(x)
            """gate_values[:, 0]: 这会返回一个一维张量（或者说向量），其形状为 (batch_size,)。
                gate_values[:, 0:1]: 这会返回一个二维张量，其形状为 (batch_size, 1)。"""
            # reward_all = torch.cat(
            #     (expert1, expert2, expert3, expert4), -1
            # )  # batch*seq_len*4
            # reward_all = torch.cat((expert1, expert2), -1)  # batch*seq_len*4
            reward_all = torch.cat(
                (expert1, expert2, expert3, expert4, expert5, expert6), -1
            )  # batch*seq_len*4
            output = (reward_all * gate_values).sum(dim=2).unsqueeze(-1)  # batch*seq
            # print("<>"*10)
            # print(output)
            # print(output.shape) #[1, 685, 1]
            return output

    model.score = MoELayer(model.score.in_features)
    
    # nn.init.xavier_uniform_(model.score.weight)
    # model.score.bias.data.fill_(0.01)

model.config.use_cache = not script_args.gradient_checkpointing
model.config.pad_token_id = tokenizer.pad_token_id
model.resize_token_embeddings(len(tokenizer))

num_proc = 24  # Can adjust to be higher if you have more processors.
# original_columns = train_dataset.column_names


# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []

        for feature in features:
            merged_features.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            merged_features.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_loss": True,
        }
        return batch


# Define the trainer
def compute_metrics(eval_pred):
    result = {}
    pos_predictions_scores = eval_pred.predictions[0]
    neg_predictions_scores = eval_pred.predictions[1]
    # We assume that the first sample is preferred by default in groundtruth
    result["accuracy"] = np.sum(pos_predictions_scores > neg_predictions_scores) / len(
        pos_predictions_scores
    )
    return result


def f_beta(x, beta):
    # eq 15 in paper "An Axiomatic Theory of Fairness in Resource Allocation"
    
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    beta = torch.tensor(beta, dtype=torch.float32)

    
    x = torch.sigmoid(x)
    
    # x = torch.exp(x)

    
    sign_term = torch.sign(1 - beta)

    
    sum_xj = torch.sum(x)

    
    ratio = x / sum_xj

    
    ratio_power = ratio ** (1 - beta)

    
    sum_term = torch.sum(ratio_power)

    
    result = sign_term * (sum_term ** (1 / beta)) / x.size(0)

    return result


class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )[0]
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]  # (bsz,1)
        rewards_k = rewards[kidx]  # (bsz,1)
        if script_args.mode == "bt":
            loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()  # (1,)
        elif script_args.mode == "fr":
            u_loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
            f_loss = -script_args.alpha * f_beta(
                rewards_j - rewards_k, script_args.beta
            )
            loss = u_loss + f_loss  # (1,)
        elif script_args.mode == "fc":
            u_loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
            f_loss = (
                f_beta(rewards_j - rewards_k, script_args.beta) ** script_args.gamma
            )
            loss = u_loss * f_loss  # (1,)
        else:
            raise ValueError(f"unsupported mode {script_args.mode}!!!")
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


# Train the model, woohoo.
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(
        tokenizer=tokenizer, max_length=script_args.max_length
    ),
)


trainer.train()


print("Saving last checkpoint of the model")
# model.save_pretrained(output_name + "/last_checkpoint")
trainer.save_model(output_name + "/last_checkpoint")
tokenizer.save_pretrained(output_name + "/last_checkpoint")
