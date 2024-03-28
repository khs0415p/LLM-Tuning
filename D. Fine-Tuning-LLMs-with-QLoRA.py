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
    HfArgumentParser,
    BitsAndBytesConfig,
    )
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model


load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    model_name: str
    device_map: str = field(default='auto')
    model_dtype: Optional[torch.dtype] = field(default=torch.bfloat16)


@dataclass
class DataArguments:
    train_data: str = field(default="drug_profile_train.json")
    valid_data: str = field(default="drug_profile_valid.json")


@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(default='results/')
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=1)
    optim: str = field(default="adamw_torch")
    warmup_steps: int = field(default=100)
    lr_scheduler_type: Optional[str] = field(default='cosine')
    fp16: bool = field(default=False)
    bf16: bool = field(default=False)
    learning_rate: float = field(default=1e-5)
    report_to: Optional[str] = field(default='none')
    logging_steps: int = field(default=10)
    gradient_accumulation_steps: int = field(default=1)
    model_max_length: int = field(default=4096)
    max_grad_norm: float = field(default=1.0)
    save_strategy: str = field(default="steps")
    save_steps: float = field(default=750)
    gradient_checkpointing: bool = field(default=True)
    bits: Optional[int] = field(default=4)
    report_to: Optional[str] = field(default='tensorboard')


class SupervisedDataset(Dataset):
    """
    데이터 형식
    [json, json, ...]
    """
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, source_prompt: str, target_prompt: str) -> None:
        super(SupervisedDataset, self).__init__()
        with open(data_path, 'r') as f:
            data = json.load(f)

        self.tokenizer = tokenizer
        self.source_template = source_prompt
        self.target_template = target_prompt

        self.sources = [self.source_template.format_map(chat_dict) for chat_dict in data]
        self.targets = [self.target_template.format_map(chat_dict) + self.tokenizer.eos_token for chat_dict in data]
            
        logging.warning("Sample data")
        logging.warning(f"source : {self.sources[0][:10]}...{self.sources[0][-10:]}")
        logging.warning(f"target : {self.targets[0][:10]}...{self.targets[0][-10:]}")

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
    logging.warning(f"Model name : {model_args.model_name} with QLoRA")
    training_args.output_dir = os.path.join(training_args.output_dir, model_args.model_name[:model_args.model_name.find('/')])
    logging.warning(f"Model name : {training_args.output_dir}-lora")
    
    if "gemma" in model_args.model_name:
        source_prompt = """<|im_start|>system\n{system_content}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>"""
        target_prompt = """<|im_start|>assistant\n{assistant_content}<|im_end|>"""
        if "it" in model_args.model_name:
            source_prompt = """<start_of_turn>user\n{system_content}\n{user_content}<end_of_turn>"""
            target_prompt = """<start_of_turn>model\n{assistant_content}<end_of_turn>"""
    elif "llama" in model_args.model_name:
        source_prompt = """[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n{user_content} [/INST]"""
        target_prompt = """ {assistant_content} """
    elif "mistral" in model_args.model_name:
        source_prompt = """[INST]{system_content}\n{user_content} [\INST]"""
        target_prompt = """{assistant_content}"""
    else:
        raise "Choose a model from gemma & llama2 & mistral"
    
    if training_args.bits == 8:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name,
            device_map = model_args.device_map,
            token=HF_TOKEN,
            # load_in_8bit=True,
            quantization_config=bnb_config
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name,
            # load_in_4bit=True,
            device_map = model_args.device_map,
            token=HF_TOKEN,
            quantization_config=bnb_config
        )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['o_proj', 'q_proj', 'up_proj', 'down_proj', 'gate_proj', 'k_proj', 'v_proj'],
        
    )

    model = get_peft_model(model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False
    )

    train_dataset = SupervisedDataset(data_path=data_args.train_data, tokenizer=tokenizer, model_name=model_args.model_name, source_prompt=source_prompt, target_prompt=target_prompt)
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
    trainer.save_state()


if __name__ == "__main__":
    train()
