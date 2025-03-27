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
from sklearn.preprocessing import StandardScaler

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer
from sentence_transformers import SentenceTransformer
import pandas as pd

# source code
from src.models.model import get_model
from transformers import DataCollatorForSeq2Seq
from src.rag.retriver import rag
from src.dataset.dataset import CustomDataset, CategoryDataset, ragdataset, SimDataset
from src.utils.metrics import compute_metrics
from src.rag.sim_prompt import sim_prompt
from src.utils.metrics import cosine_similarity, jaccard_similarity

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from typing import Dict, Any

def run(config: Dict[str, Any]) -> float:

    dir = config['train']['parameter_save'] + '/' + config['train']['test']['dir']
    print(dir)
    
    embedding_model_name = "jhgan/ko-sbert-sts"
    embedding = SentenceTransformer(embedding_model_name)

    temp = sim_prompt(config['data'])

    test = pd.read_csv(config['data']['dir'] + config['data']['test_csv'], encoding = 'utf-8-sig')
    question = test['사고원인']

    test_results = []
    for q in tqdm(question, total=len(question), desc='question'):
        prompt = [temp.get_value(k.page_content) for k in temp.get_answer(q)]
        
        q_embd = embedding.encode(q)
        embeddings = embedding.encode(prompt)

        best_answer = None
        best_score = -1

        for i, p in zip(embeddings, prompt):
            cosine = cosine_similarity(q_embd, i).item()

            jaccard_sim = jaccard_similarity(q, p)

            total_score = (0.7 * cosine + 0.3 * jaccard_sim)
            if total_score > best_score:
                best_answer = p
                best_score = total_score

        test_results.append(best_answer)

    train_df = pd.read_csv(config['data']['dir'] + config['data']['train_csv'])

    prevention_vectors = embedding.encode(train_df["재발방지대책 및 향후조치계획"].tolist(), convert_to_numpy=True, show_progress_bar=False)

    mean_vector = np.mean(prevention_vectors, axis=0)

    mutate_vector = []

    for t in tqdm(test_results, total=len(test_results), desc="mutate"):
        embd = embedding.encode(t, convert_to_numpy=True, show_progress_bar=False)

        direction = embd - mean_vector
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm == 0:
            mutated = mean_vector
        else:
            max_norm = np.linalg.norm(mean_vector)
            step_size = 0.2
            new_vec = mean_vector + step_size * direction
            
            new_norm = np.linalg.norm(new_vec)
            if new_norm > max_norm:
                new_vec = new_vec / new_norm * max_norm
            
            mutated = new_vec

        mutate_vector.append(mutated)

    test_results = ["작업전 안전교육 강화 및 작업장 위험요소 점검을 통한 재발 방지와 안전관리 교육 철저를 통한 향후 조치 계획." for _ in range(len(test_results))]

    submission = pd.read_csv(f"{config['data']['dir']}{config['data']['submission_csv']}", encoding="utf-8-sig")

    submission.iloc[:, 1] = test_results  
    submission.iloc[:, 2:] = mutate_vector  

    submission.to_csv(f"{config['data']['result_dir']}{config['data']['result_file']}", index=False, encoding="utf-8-sig")

    print("✅ 최종 결과 저장 완료!")
