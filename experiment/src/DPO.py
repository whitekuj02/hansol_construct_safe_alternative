from typing import Any, Dict
import os
import random
import yaml
import argparse
import glob
import pandas as pd
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from datasets import Dataset
from transformers import TrainingArguments, Trainer
from trl import SFTTrainer
from trl import DPOTrainer
from trl import DPOConfig

from src.models.model import get_model
from src.dataset.dataset import CustomDataset, CategoryDataset, ragdataset
from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback
from src.rag.retriver import rag
from src.utils.metrics import compute_metrics
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset as HFDataset


def run(config: Dict[str, Any]) -> float:
    print("1. 학습한 SFT 모델 불러오기")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dir = config['train']['parameter_save'] + '/' + config['train']['test']['dir']

    model = AutoModelForCausalLM.from_pretrained(dir, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(dir)

    sft_model_dir = config['train']['parameter_save'] + '/' + config['train']['test']['dir']
    print(f"sft_model_dir: {sft_model_dir}")

    print(f"학습한 SFT 모델 dir : {sft_model_dir}")

    if tokenizer.pad_token is None : 
        tokenizer.pad_token = tokenizer.eos_token 

    
    csv_path = "/root/construct/experiment/result/dpo_data.csv"
    raw_dict = {}

    if os.path.exists(csv_path):
        print(f"기존 CSV 파일을 사용하여 DPO 데이터셋을 생성합니다: {csv_path}")
        df = pd.read_csv(csv_path)
        raw_dict = {
            "prompt": df["prompt"].tolist(),
            "chosen": df["chosen"].tolist(),
            "rejected": df["rejected"].tolist(),
        }
    else:
        print("2. 학습한 모델로 train.csv 추론, rejected 추출하기")
        if config['data']['rag']['use']:
            dataset = ragdataset(config['data'], mode='train')
        else:
            dataset = CustomDataset(config['data'], mode='train')
        data_items = dataset
        total_items = len(data_items)

        train_dir = config['data']['dir'] + config['data']['train_csv']
        train = pd.read_csv(train_dir)
        train_length = len(train)
        divide = (total_items // train_length)
        n_rows = total_items // divide
        print(f"Flatten된 데이터 개수: {total_items}, 원본 행 개수: {n_rows}")

        rejected_data = []
        model.to(device)
        
        for i in tqdm(range(n_rows), desc="Generating Rejected"):
            group_5 = data_items[i * divide: i * divide + divide]
            chosen_item = random.choice(group_5)

            question = chosen_item["question"]
            choosen_answer = chosen_item["answer"]

            if config["data"]['rag']['use']:
                retriver = chosen_item['retriver']
                input_text = f"""다음은 질문과 관련된 정보들입니다.
{retriver}
위의 정보들을 참고하여 답변 생성 시 정보와 질문의 내용에 관련 내용이 있다면 반드시 반영해 주시기 바랍니다.
질문: {question}
대답:"""
            else:
                input_text = f"질문: {question}대답:"

            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                model_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                rejected_answer = model_output.split("대답: ")[-1].strip()

                rejected_data.append({
                    "prompt": question,
                    "chosen": choosen_answer,
                    "rejected": rejected_answer
                })

            del inputs, output_ids
            torch.cuda.empty_cache()

        raw_dict = {
            "prompt": [rd["prompt"] for rd in rejected_data],
            "chosen": [rd["chosen"] for rd in rejected_data],
            "rejected": [rd["rejected"] for rd in rejected_data],
        } 

        df = pd.DataFrame(raw_dict)
        df.to_csv(csv_path, index=False)
        print(f"새로운 DPO 데이터 CSV 파일 생성 완료: {csv_path}")

    dpo_dataset = HFDataset.from_dict(raw_dict)

    print("4. DPO training")

    dpo_args = DPOConfig(
        **config["train"]["dpo"]
    )

    ref_model = AutoModelForCausalLM.from_pretrained(sft_model_dir, torch_dtype=torch.bfloat16).to(device)

    torch.cuda.empty_cache()
    model.train()

    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    ref_model.eval()

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=dpo_dataset,
        processing_class=tokenizer
    )

    dpo_trainer.train()
    dpo_trainer.save_model(dpo_args.output_dir)

    print(f"DPO 모델 저장 완료: {dpo_args.output_dir}")
    return 0
