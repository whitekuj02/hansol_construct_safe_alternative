from typing import Any, Dict
import os
import random
import yaml
import argparse
import glob
import re
# library
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from datasets import Dataset
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.llms import HuggingFacePipeline

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer
import pandas as pd

# source code
from src.models.model import get_model
from transformers import DataCollatorForSeq2Seq
from src.rag.retriver import rag
from src.dataset.dataset import CustomDataset, CategoryDataset, ragdataset, SimDataset
from src.utils.metrics import compute_metrics

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from typing import Dict, Any

def run(config: Dict[str, Any]) -> float:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dir = config['train']['parameter_save'] + '/' + config['train']['test']['dir']
    #dir = "/root/construct/experiment/dpo_model/checkpoint-23422"
    
    print(dir)
    model = AutoModelForCausalLM.from_pretrained(dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(dir)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"모델 설정 확인 : {model.config}")
    
    if config['data']['category']:
        test_dataset = SimDataset(config['data'], mode="test").to_huggingface_dataset()
    elif config['data']['rag']['use']:
        test_dataset = ragdataset(config['data'], mode="test").to_huggingface_dataset()
    else:
        test_dataset = CustomDataset(config['data'], mode="test").to_huggingface_dataset()

    print("테스트 실행 시작... 총 테스트 샘플 수:", len(test_dataset))

    test_results = []

    for idx, row in enumerate(test_dataset):
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"\n[샘플 {idx + 1}/{len(test_dataset)}] 진행 중...")
        
        if config['data']['category']:
            question = f"""다음은 사건의 대표 답변입니다:
{row['retriver']}
위의 대표 답변들을 참고하여, 아래 질문에 대해 적절한 답변을 생성해 주세요.
질문: {row['question']}
대답:"""
        elif config['data']['rag']['use']:
            question = f"""다음은 질문과 관련된 정보들입니다.
{row['retriver']}
위의 정보들을 참고하여 답변 생성 시 정보와 질문의 내용에 관련 내용이 있다면 반드시 반영해 주시기 바랍니다.
질문: {row['question']}
대답:"""
        else:
            question = f"질문: {row['question']}대답:"

        input_ids = tokenizer(
            question, 
            truncation=True,  
            max_length=512,   
            return_tensors="pt",  
            padding=False
        ).to(device)

        output_ids = model.generate(
            **input_ids,
            max_new_tokens=64,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        answer = answer.split("대답: ")[-1]

        if answer == "":
            answer = "작업전 안전교육 강화 및 작업장 위험요소 점검을 통한 재발 방지와 안전관리 교육 철저를 통한 향후 조치 계획."
        
        print(answer)
        test_results.append(answer)

    print("\n테스트 실행 완료! 총 결과 수:", len(test_results))

    embedding_model_name = "jhgan/ko-sbert-sts"
    embedding = SentenceTransformer(embedding_model_name)

    pred_embeddings = embedding.encode(test_results)
    print(pred_embeddings.shape)

    submission = pd.read_csv(f"{config['data']['dir']}{config['data']['submission_csv']}", encoding="utf-8-sig")

    submission.iloc[:, 1] = test_results
    submission.iloc[:, 2:] = pred_embeddings

    submission.to_csv(f"{config['data']['result_dir']}{config['data']['result_file']}", index=False, encoding="utf-8-sig")

    print("✅ 최종 결과 저장 완료!")
