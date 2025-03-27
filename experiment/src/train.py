from typing import Any, Dict
import os
import random
import yaml
import argparse
import glob

# library
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from trl import SFTTrainer
from peft import PeftModel

# source code
from src.models.model import get_model
# from src.dataset.dataloader import get_train_data_loaders, get_test_data_loaders
from src.dataset.dataset import CustomDataset, CategoryDataset, ragdataset, SimDataset
from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback
from src.rag.retriver import rag
from src.utils.metrics import compute_metrics

def run(config: Dict[str, Any]) -> float:
    device = "cuda" if (config.get('device') == 'cuda' and torch.cuda.is_available()) else "cpu"

    if config['train'].get('resume', False):
        sft_model_dir = config['train']['resume_checkpoint']
        print(f"[INFO] Loading SFT model from: {sft_model_dir}")

        model = AutoModelForCausalLM.from_pretrained(sft_model_dir, torch_dtype=torch.float16).to(device)
        tokenizer = AutoTokenizer.from_pretrained(sft_model_dir)

        torch.cuda.empty_cache()

        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        
        model.train() 
    else:
        print("[INFO] Starting training from scratch (no LoRA resume).")
        tokenizer, model = get_model(config['train']['model'])
        model = model.to(device)


    if config['data']['category']:
        train_dataset = SimDataset(config['data'], mode="train").to_huggingface_dataset()
    elif config['data']['rag']['use']:
        train_dataset = ragdataset(config['data'], mode="train").to_huggingface_dataset()
    else:
        train_dataset = CustomDataset(config['data'], mode="train").to_huggingface_dataset()

    print("데이터 불러오기 성공")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    def category_preprocess_function(examples):
        system_prompt = """다음은 사건의 대표 답변입니다:
{0}
위의 대표 답변들을 참고하여, 아래 질문에 대해 적절한 답변을 생성해 주세요.
{1}
{2}"""
        instructions = [f"질문: {q}" for q in examples["question"]]
        retrivers =  [f"{q}" for q in examples["retriver"]]
        responses = [f"대답: {a}" for a in examples["answer"]]
        full_texts = [system_prompt.format(retr, inst, resp) for retr, inst, resp in zip(retrivers, instructions, responses)]

        #print(full_texts[0])
        model_inputs = tokenizer(full_texts, padding="max_length", truncation=True, max_length=512)

        input_ids = model_inputs["input_ids"] 
        labels = [ids.copy() for ids in input_ids] 

        for i in range(len(labels)):
            except_answer = full_texts[i].split("대답: ")[0]
            question_ids = tokenizer(f"{except_answer}대답: ", add_special_tokens=False)["input_ids"]
            question_len = len(question_ids)

            labels[i][:question_len] = [-100] * min(question_len, len(labels[i]))

            eos_idx = [idx for idx, token in enumerate(labels[i]) if token == tokenizer.pad_token_id]

            last_pad_idx = min(eos_idx) if eos_idx else 0 

            labels[i] = [
                -100 if (idx > last_pad_idx and token == tokenizer.pad_token_id) else token
                for idx, token in enumerate(labels[i])
            ]

            labels[i] = [
                tokenizer.eos_token_id if token == tokenizer.pad_token_id else token
                for token in labels[i]
            ]

        model_inputs["labels"] = labels
        return model_inputs
    
    def preprocess_function(examples):
        instructions = [f"질문: {q} " for q in examples["question"]]
        responses = [f"대답: {a}" for a in examples["answer"]]
        full_texts = [inst + resp for inst, resp in zip(instructions, responses)]

        model_inputs = tokenizer(full_texts, padding="max_length", truncation=True, max_length=512)

        input_ids = model_inputs["input_ids"] 
        labels = [ids.copy() for ids in input_ids] 

        for i in range(len(labels)):
            except_answer = full_texts[i].split("대답: ")[0]
            question_ids = tokenizer(f"{except_answer}대답: ", add_special_tokens=False)["input_ids"]
            question_len = len(question_ids)

            labels[i][:question_len] = [-100] * min(question_len, len(labels[i]))

            # labels[i] = [-100 if token == tokenizer.pad_token_id else token for token in labels[i]]

            eos_idx = [idx for idx, token in enumerate(labels[i]) if token == tokenizer.pad_token_id]

            last_pad_idx = min(eos_idx) if eos_idx else 0 

            labels[i] = [
                -100 if (idx > last_pad_idx and token == tokenizer.pad_token_id) else token
                for idx, token in enumerate(labels[i])
            ]

            labels[i] = [
                tokenizer.eos_token_id if token == tokenizer.pad_token_id else token
                for token in labels[i]
            ]

        model_inputs["labels"] = labels
        return model_inputs
    
    def rag_preprocess_function(examples):
        system_prompt = """다음은 질문과 관련된 정보들입니다.
{0}
위의 정보들을 참고하여 답변 생성 시 정보와 질문의 내용에 관련 내용이 있다면 반드시 반영해 주시기 바랍니다.
{1}
{2}"""
        instructions = [f"질문: {q}" for q in examples["question"]]
        retrivers =  [f"{q}" for q in examples["retriver"]]
        responses = [f"대답: {a}" for a in examples["answer"]]
        full_texts = [system_prompt.format(retr, inst, resp) for retr, inst, resp in zip(retrivers, instructions, responses)]

        model_inputs = tokenizer(full_texts, padding="max_length", truncation=True, max_length=512)

        input_ids = model_inputs["input_ids"] 
        labels = [ids.copy() for ids in input_ids] 

        for i in range(len(labels)):
            except_answer = full_texts[i].split("대답: ")[0]
            question_ids = tokenizer(f"{except_answer}대답: ", add_special_tokens=False)["input_ids"]
            question_len = len(question_ids)

            labels[i][:question_len] = [-100] * min(question_len, len(labels[i]))

            eos_idx = [idx for idx, token in enumerate(labels[i]) if token == tokenizer.pad_token_id]

            last_pad_idx = min(eos_idx) if eos_idx else 0 

            labels[i] = [
                -100 if (idx > last_pad_idx and token == tokenizer.pad_token_id) else token
                for idx, token in enumerate(labels[i])
            ]

            labels[i] = [
                tokenizer.eos_token_id if token == tokenizer.pad_token_id else token
                for token in labels[i]
            ]

        model_inputs["labels"] = labels
        return model_inputs
    
    if config['data']['category']:
        train_dataset = train_dataset.map(category_preprocess_function, batched=True)
    elif config['data']['rag']['use']:
        train_dataset = train_dataset.map(rag_preprocess_function, batched=True)
    else:
        train_dataset = train_dataset.map(preprocess_function, batched=True)
    
    print(tokenizer.pad_token_id, tokenizer.eos_token_id)
    print(train_dataset[0])

    training_args = TrainingArguments(
        **config['train']['training']
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, max_length=512)

    if config['train']['SFT']:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator
        )

    if config['train'].get('resume', False):
        trainer.train(resume_from_checkpoint=config['train']['resume_checkpoint'])
    else:
        trainer.train()


    dir = config['train']['parameter_save'] + '/' + config['train']['test']['dir']
    os.makedirs(dir, exist_ok=True)
    trainer.args.output_dir = dir
    trainer.save_model(dir)
    trainer.save_state()
    tokenizer.save_pretrained(dir)



if __name__ == "__main__":
     
    def get_config(config_folder):
        config = {}

        config_folder = os.path.join(config_folder,'*.yaml')
        
        config_files = glob.glob(config_folder)

        for file in config_files:
            with open(file, 'r') as f:
                config.update(yaml.safe_load(f))
        
        if config['device'] == 'cuda' and not torch.cuda.is_available():
            print('using cpu now...')
            config['device'] = 'cpu'

        return config
    
    config = get_config("/home/aicontest/construct/experiment/configs")
    run(config)