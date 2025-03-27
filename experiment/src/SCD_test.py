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
from sentence_transformers import SentenceTransformer, util
import pandas as pd

# source code
from src.models.model import get_model
from transformers import DataCollatorForSeq2Seq
from src.rag.retriver import rag
from src.dataset.dataset import CustomDataset, CategoryDataset, ragdataset
from src.utils.metrics import compute_metrics
from src.utils.metrics import cosine_similarity, jaccard_similarity

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
from datasets import Dataset
from typing import Dict, Any

def run(config: Dict[str, Any]) -> float:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dir = config['train']['parameter_save'] + '/' + config['train']['test']['dir']
    print(dir)
    # 모델과 토크나이저 로드
    model = AutoModelForCausalLM.from_pretrained(dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(dir)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"모델 설정 확인 : {model.config}")
    
    # 테스트 데이터 로드
    if config['data']['category']:
        test_dataset = CategoryDataset(config['data'], mode="SCD").to_huggingface_dataset()
    elif config['data']['rag']['use']:
        test_dataset = ragdataset(config['data'], mode="SCD").to_huggingface_dataset()
    else:
        test_dataset = CustomDataset(config['data'], mode="SCD").to_huggingface_dataset()

    print("테스트 실행 시작... 총 테스트 샘플 수:", len(test_dataset))

    total_items = len(test_dataset)
    train_dir = config['data']['dir'] + config['data']['test_csv']
    test = pd.read_csv(train_dir)
    test_length = len(test)
    divide = (total_items // test_length)
    n_rows = total_items // divide

    # 결과 저장 리스트
    test_results = []

    # 모델 추론 실행
    for i in tqdm(range(n_rows), desc="Generating SCD"):
        group_5 = test_dataset[i * divide: i * divide + divide]
        
        if config['data']['category']:
            question = """다음은 사건의 특징 별로 추출된 대표 답변 데이터입니다:
{0}
위의 대표 답변들을 참고하여, 아래 질문에 대해 적절한 답변을 생성해 주세요. 답변 생성 시, 대표 답변이 질문의 내용과 관련이 있다면 반드시 반영해 주시기 바랍니다.
질문: {1}
대답:"""
            question_5 = [question.format(r, q) for r, q in zip(group_5['retriver'], group_5['question'])]
        elif config['data']['rag']['use']:
            question = """다음은 질문과 관련된 정보들입니다.
{0}
위의 정보들을 참고하여 답변 생성 시 정보와 질문의 내용에 관련 내용이 있다면 반드시 반영해 주시기 바랍니다.
질문: {1}
대답:"""
            question_5 = [question.format(r, q) for r, q in zip(group_5['retriver'], group_5['question'])]
        else:
            question = "질문: {0}대답:"
            question_5 = [question.format(i) for i in group_5['question']]

        embedding_model_name = "jhgan/ko-sbert-sts"
        embedding = SentenceTransformer(embedding_model_name)

        output_list = []
        for idx, i in enumerate(question_5):
            input_ids = tokenizer(
                i, 
                truncation=True,  # 길이가 max_length보다 길면 잘라냄
                max_length=512,   # 최대 길이 제한
                return_tensors="pt",  
                padding=False  # 패딩 제거 (또는 생략 가능)
            ).to(device)

            output_ids = model.generate(
                **input_ids,  # 질문을 그대로 입력
                max_new_tokens=64,  # 답변 길이만 제한
                do_sample=False,  # 확률적 샘플링 비활성화 (일관된 결과 보장)
                num_beams=1,  # Greedy Decoding 사용 (최적의 답변 출력)
                repetition_penalty=1.2,
                pad_token_id=tokenizer.pad_token_id,  # 패딩을 EOS로 설정
                eos_token_id=tokenizer.eos_token_id  # EOS에서 멈추도록 설정
            )

            answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            answer = answer.split("대답: ")[-1]

            if answer == "":
                answer = "재발 방지 대책 및 향후 조치 계획"
            
            print(f" 추론 {idx}: {answer}")
            output_list.append(answer)

        embeddings = embedding.encode(output_list, convert_to_tensor=True)
        question = embedding.encode(question_5, convert_to_tensor=True)

        # 가장 의미적으로 일관된 답변 선택
        best_answer = None
        best_score = -1

        # for i, answer1 in enumerate(output_list):
        #     total_score = 0
        #     for j, answer2 in enumerate(output_list):
        #         if i != j:
        #             # 코사인 유사도
        #             cosine_sim = util.pytorch_cos_sim(embeddings[i], embeddings[j]).item()

        #             # 자카드 유사도
        #             jaccard_sim = jaccard_similarity(answer1, answer2)

        #             # 최종 유사도: 0.7 * 코사인 유사도 + 0.3 * 자카드 유사도
        #             total_score += (0.7 * cosine_sim + 0.3 * jaccard_sim)

        #     # 평균 유사도가 가장 높은 문장을 선택
        #     avg_score = total_score / (len(output_list) - 1)  # 자기 자신 제외
        #     if avg_score > best_score:
        #         best_score = avg_score
        #         best_answer = answer1

        for q, a, t in zip(question, embeddings, output_list):
            cosine_sim = util.pytorch_cos_sim(q, a).item()

            if best_score < cosine_sim:
                best_score = cosine_sim
                best_answer = t

        print(f"SCD 결과 : {best_answer}")
        test_results.append(best_answer)



    print("\n테스트 실행 완료! 총 결과 수:", len(test_results))


    pred_embeddings = embedding.encode(test_results)
    print(pred_embeddings.shape)  # (샘플 개수, 768)

    # ✅ 최종 결과 CSV 저장
    submission = pd.read_csv(f"{config['data']['dir']}{config['data']['submission_csv']}", encoding="utf-8-sig")

    submission.iloc[:, 1] = test_results  # 모델 생성된 답변 추가
    submission.iloc[:, 2:] = pred_embeddings  # 생성된 문장 임베딩 추가

    submission.to_csv(f"{config['data']['result_dir']}{config['data']['result_file']}", index=False, encoding="utf-8-sig")

    print("✅ 최종 결과 저장 완료!")
