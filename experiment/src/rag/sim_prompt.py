# question 에 대한 유사한 prompt 
# sim_prompt
import re
import os
import itertools
import numpy as np
from typing import Any, Dict
import random
import yaml
import argparse
import glob
import torch

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores.utils import DistanceStrategy
import faiss
import itertools

import numpy as np
import pandas as pd 
from typing import Any, Dict
import random

from tqdm import tqdm
# pandas의 apply와 함께 진행 상황 표시를 위해 필요
tqdm.pandas()

class sim_prompt:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

        self.dir = config['dir'] + config['train_csv']
        self.data_df = pd.read_csv(self.dir, encoding='utf-8-sig')

        self.dict = {}
        self.data_df['장소(대)'] = self.data_df['장소'].str.split(' / ').str[0].fillna("")
        self.data_df['장소(소)'] = self.data_df['장소'].str.split(' / ').str[1].fillna("")
        self.data_df['공사종류(대분류)'] = self.data_df['공사종류'].str.split(' / ').str[0].fillna("")
        self.data_df['공사종류(중분류)'] = self.data_df['공사종류'].str.split(' / ').str[1].fillna("")
        self.data_df['공종(대분류)'] = self.data_df['공종'].str.split(' > ').str[0].fillna("")
        self.data_df['공종(중분류)'] = self.data_df['공종'].str.split(' > ').str[1].fillna("")
        self.data_df['사고객체(대분류)'] = self.data_df['사고객체'].str.split(' > ').str[0].fillna("")
        self.data_df['사고객체(중분류)'] = self.data_df['사고객체'].str.split(' > ').str[1].fillna("")
        self.data_df['부위(대)'] = self.data_df['부위'].str.split(' / ').str[0].fillna("")
        self.data_df['부위(소)'] = self.data_df['부위'].str.split(' / ').str[1].fillna("")

        def make_prompts_for_row(row):
            question = (
                f"{row['사고원인']}"
            )

            answer_text = str(row.get('재발방지대책 및 향후조치계획', ''))
            self.dict[question] = answer_text

        self.data_df.progress_apply(make_prompts_for_row, axis=1)
        print(f"전체 데이터 길이: {len(self.data_df)}")

        # key
        self.keys = list(self.dict.keys())
        print(f"keys 갯수 : {len(self.keys)}")

        self.embedding_model_name = config['rag']['embedding_model']  # 임베딩 모델 선택
        self.embedding = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

        self.vector_store = FAISS.from_texts(self.keys, self.embedding, distance_strategy = DistanceStrategy.COSINE)

        self.retriever = self.vector_store.as_retriever(search_type=config['rag']['search_type'], search_kwargs={**config['rag']['search_kwargs']})
        

    def get_value(self, prompt): #prompt -> dict -> value 
        return self.dict[prompt]

    def get_answer(self, query): # test.csv question -> rag -> prompt
        return self.retriever.get_relevant_documents(query)

def cosine_similarity(a, b):
    """코사인 유사도 계산"""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if norm_a != 0 and norm_b != 0 else 0


def jaccard_similarity(text1, text2):
    """자카드 유사도 계산"""
    set1, set2 = set(text1.split()), set(text2.split())  # 단어 집합 생성
    intersection = len(set1.intersection(set2))  # 교집합 크기
    union = len(set1.union(set2))  # 합집합 크기
    return intersection / union if union != 0 else 0


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
    
    config = get_config("/root/construct/experiment/configs")
    temp = sim_prompt(config['data'])
    
    embedding_model_name = "jhgan/ko-sbert-sts"
    embedding = SentenceTransformer(embedding_model_name)
    
    test = pd.read_csv(config['data']['dir'] + config['data']['test_csv'], encoding = 'utf-8-sig')
    question = test['사고원인']

    test_results = []
    total_count = 0
    for q in tqdm(question, total=len(question), desc='question'):
        prompt = [temp.get_value(k.page_content) for k in temp.get_answer(q)]

        #print("\n".join(prompt))
        q_embd = embedding.encode(q)
        embeddings = embedding.encode(prompt)

        ALL_THRESHOLD = 0.5 # 평균은 0.35
        total = 0
        sim = []
        for i in embeddings:
            cosine = cosine_similarity(q_embd, i).item()
            sim.append(cosine)
            total += cosine

        best_answer = None
        best_score = -1

        for i, answer1 in enumerate(prompt):
            total_score = 0
            for j, answer2 in enumerate(prompt):
                if i != j:
                    # 코사인 유사도
                    cosine_sim = cosine_similarity(embeddings[i], embeddings[j]).item()

                    # 최종 유사도: 0.7 * 코사인 유사도 + 0.3 * 자카드 유사도
                    total_score += cosine_sim

            # 평균 유사도가 가장 높은 문장을 선택
            avg_score = total_score / (len(prompt) - 1)  # 자기 자신 제외
            if avg_score > best_score:
                best_score = avg_score
                best_answer = answer1

        test_results.append(best_answer)

    print(f"전체 답안 변환 비율 : {total_count / len(question)}")
    pred_embeddings = embedding.encode(test_results)
    print(pred_embeddings.shape)  # (샘플 개수, 768)

    # ✅ 최종 결과 CSV 저장
    submission = pd.read_csv(f"{config['data']['dir']}{config['data']['submission_csv']}", encoding="utf-8-sig")

    submission.iloc[:, 1] = test_results  # 모델 생성된 답변 추가
    submission.iloc[:, 2:] = pred_embeddings  # 생성된 문장 임베딩 추가

    submission.to_csv(f"{config['data']['result_dir']}{config['data']['result_file']}", index=False, encoding="utf-8-sig")

    print("✅ 최종 결과 저장 완료!")
