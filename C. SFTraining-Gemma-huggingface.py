import os
import json
import copy
import torch
import logging

from tqdm import tqdm
from dotenv import load_dotenv
from torch.utils.data import Dataset
from typing import Optional, Sequence, Dict
from dataclasses import dataclass, field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser
    )

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
IGNORE_INDEX = -100
SOURCE_PROMPT = """<|im_start|>system\n{system_content}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n"""
TARGET_PROMPT = """<|im_start|>assistant\n{assistant_content}<|im_end|>\n"""


@dataclass
class ModelArguments:
    model_name: Optional[str] = field(default="google/gemma-2b")
    device_map: str = field(default='auto')


@dataclass
class DataArguments:
    train_data: str = field(default="YOUR_DATA.json")
    valid_data: str = field(default="YOUR_DATA.json")


@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(default='results/')
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=1)
    optim: str = field(default="adamw_torch")
    warmup_steps: int = field(default=2)
    lr_scheduler_type: Optional[str] = field(default='cosine')
    fp16: bool = field(default=True)
    bf16: bool = field(default=False)
    learning_rate: float = field(default=1e-5)
    report_to: Optional[str] = field(default='none')
    logging_steps: int = field(default=10)
    gradient_accumulation_steps: int = field(default=1)
    model_max_length: int = field(default=2048)
    max_grad_norm: float = field(default=1.0)
    save_strategy: str = field(default="epoch")


class SupervisedDataset(Dataset):
    """
    데이터 형식
    [json, json, ...]
    """
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer) -> None:
        super(SupervisedDataset, self).__init__()
        with open(data_path, 'r') as f:
            data = json.load(f)

        self.tokenizer = tokenizer
        self.sources = [SOURCE_PROMPT.format_map(chat_dict) for chat_dict in data]
        self.targets = [TARGET_PROMPT.format_map(chat_dict) for chat_dict in data]
        logging.warning("Sample data")
        logging.warning(f"source : {self.sources[0][:50]}...")
        logging.warning(f"source : {self.targets[0][:50]}...")

        self.preprocess()


    def _tokenize(self, sequences: Sequence[str]):
        tokenized_list = [
            self.tokenizer(
                seq,
                return_tensors='pt',
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                padding="longest"
            )
            for seq in sequences
        ]
        input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = [tokenized.input_ids.ne(self.tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
        logging.warning(f"- Max length : {max(input_ids_lens)}")

        return {
            "input_ids": input_ids,
            "input_ids_lens": input_ids_lens
        }

    def preprocess(self):
        full_strings = [src + trg for src, trg in zip(self.sources, self.targets)]
        full_tokenized, source_tokenized = [self._tokenize(sequences) for sequences in (tqdm(full_strings), tqdm(self.sources))]
        input_ids = full_tokenized['input_ids']
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, source_tokenized['input_ids_lens']):
            label[:source_len] = IGNORE_INDEX

        self.input_ids = input_ids
        self.labels = labels        

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "labels": self.labels[index]
        }


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return {
            "input_ids" : input_ids,
            "labels" : labels,
            "attention_mask" : input_ids.ne(self.tokenizer.pad_token_id),
        }


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name,
        device_map = model_args.device_map,
        token=HF_TOKEN,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False
    )

    tokenizer.add_tokens(["<|im_start|>"])
    tokenizer.add_special_tokens({"eos_token": "<|im_end|>"})

    # resize model
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = SupervisedDataset(data_path=data_args.train_data, tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
        )
    model.config.use_cache = False

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    train()
