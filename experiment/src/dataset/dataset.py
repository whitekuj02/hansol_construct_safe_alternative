from typing import Any, Dict
import os
import random
import yaml
import argparse
import glob

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

import pandas as pd
from datasets import Dataset as HFDataset
from typing import Dict, Any, List, Union
from ..utils.metrics import cosine_similarity
from sentence_transformers import SentenceTransformer
from ..rag.retriver import rag
from ..rag.sim_prompt import sim_prompt

def category_similarity(train, model, col):
    grouped = train.groupby(col)
    res = {}
    for name, group in tqdm(grouped, desc=col):
        plan = group["재발방지대책 및 향후조치계획"]
        vectors = np.stack(plan.apply(model.encode).to_numpy())
        similarity = cosine_similarity(vectors, vectors)
        res[name] = plan.iloc[similarity.mean(axis=1).argmax()]
    return res


class CategoryDataset(Dataset):
    """
    기존 CategoryDataset은 그대로 두었습니다.
    """
    def __init__(self, config: Dict[str, Any], mode: str):
        self.data_config = config
        self.data_dir = config['dir']
        self.mode = mode

        # 데이터 경로 설정
        if mode == "train":
            data_path = self.data_dir + config['train_csv']
        elif mode == "test" or mode == "SCD":
            data_path = self.data_dir + config['test_csv']
        else:
            raise ValueError("mode는 'train' 또는 'test'만 가능합니다.")

        self.model = SentenceTransformer('jhgan/ko-sbert-sts', use_auth_token=False)
        # CSV 파일 로드
        self.data_df = pd.read_csv(data_path, encoding='utf-8-sig').fillna("없음")

        self.columns = ["인적사고", "사고객체", "작업프로세스"]
        self.column_most = {}

        self.retriver_path = self.data_dir + config['train_csv']
        self.retriver_df = pd.read_csv(self.retriver_path, encoding='utf-8-sig').fillna("없음")
        for i in self.columns:
            res = category_similarity(self.retriver_df, self.model, i)
            self.column_most[i] = res

        
        self.data_df['공사종류(대분류)'] = self.data_df['공사종류'].str.split(' / ').str[0].fillna("")
        self.data_df['공사종류(중분류)'] = self.data_df['공사종류'].str.split(' / ').str[1].fillna("")
        self.data_df['공종(대분류)'] = self.data_df['공종'].str.split(' > ').str[0].fillna("")
        self.data_df['공종(중분류)'] = self.data_df['공종'].str.split(' > ').str[1].fillna("")
        self.data_df['사고객체(대분류)'] = self.data_df['사고객체'].str.split(' > ').str[0].fillna("")
        self.data_df['사고객체(중분류)'] = self.data_df['사고객체'].str.split(' > ').str[1].fillna("")

        # 질문 생성
        def format_prompt(row):
            # 공사종류 = self.column_most['공사종류'][row['공사종류']]
            인적사고 = self.column_most['인적사고'][row['인적사고']]
            # 공종 = self.column_most['공종'][row['공종']]
            사고객체 = self.column_most['사고객체'][row['사고객체']]
            작업프로세스 = self.column_most['작업프로세스'][row['작업프로세스']]
            사고원인 = row['사고원인']

            retriver = f"{인적사고},{사고객체},{작업프로세스}"

            question1 = (
                f"작업 '{row['작업프로세스']}' 도중 '{row['인적사고']}' 사고가 발생했다. "
                + f"사고 객체: {row['사고객체(대분류)']}({row['사고객체(중분류)']}). "
                + f"이 때 사고 원인은 '{row['사고원인']}'이다."
                + "예방 방안은 무엇인가?"
            )

            question2 = (
                f"{row['작업프로세스']} 도중 '{row['인적사고']}' 사고가 났다. "
                + f"사고 객체: {row['사고객체(대분류)']}({row['사고객체(중분류)']}). "
                + f"이 때 사고 원인은 '{row['사고원인']}'이다."
                + "어떤 조치를 취해야 하는가?"
            )

            question3 = (
                f"작업프로세스 '{row['작업프로세스']}', 인적사고: '{row['인적사고']}', "
                + f"사고 객체: {row['사고객체(대분류)']}({row['사고객체(중분류)']}). "
                + f"이 때 사고 원인은 '{row['사고원인']}'이다."
                + "대응책은 무엇인가?"
            )

            if self.mode == "train":
                answer = f"{row['재발방지대책 및 향후조치계획']}"
            else:  # test 모드에서는 정답 없음
                answer = ""

            return [
                {"retriver": retriver, "question": question1, "answer": answer},
                {"retriver": retriver, "question": question2, "answer": answer},
                {"retriver": retriver, "question": question3, "answer": answer}
            ]

        all_rows_2d = self.data_df.apply(format_prompt, axis=1).tolist()
        
        # Flatten 처리
        if self.mode == "train" or mode == "SCD":
            flatten_data = []
            for row_prompts in all_rows_2d:
                flatten_data.extend(row_prompts)

            # 최종적으로 self.data에 할당
            self.data = flatten_data
        else : # test 일때는 1개만 처리 
            self.data = [random.choice(row_prompts) for row_prompts in all_rows_2d]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def to_huggingface_dataset(self):
        return HFDataset.from_list(self.data)

class SimDataset(Dataset):
    """
    기존 CategoryDataset은 그대로 두었습니다.
    """
    def __init__(self, config: Dict[str, Any], mode: str):
        self.data_config = config
        self.data_dir = config['dir']
        self.mode = mode

        # 데이터 경로 설정
        if mode == "train":
            data_path = self.data_dir + config['train_csv']
        elif mode == "test" or mode == "SCD":
            data_path = self.data_dir + config['test_csv']
        else:
            raise ValueError("mode는 'train' 또는 'test'만 가능합니다.")

        self.model = SentenceTransformer('jhgan/ko-sbert-sts')
        # CSV 파일 로드
        self.data_df = pd.read_csv(data_path, encoding='utf-8-sig').fillna("없음")

        temp = sim_prompt(config)
        question = self.data_df['사고원인'].tolist()
        
        retriver = []
        for q in tqdm(question, total=len(question), desc='question'):
            prompt = [temp.get_value(k.page_content) for k in temp.get_answer(q)]

            if mode == "train":
                prompt = prompt[1:]
            #print("\n".join(prompt))
            q_embd = self.model.encode(q)
            embeddings = self.model.encode(prompt)

            ALL_THRESHOLD = 0.5 # 평균은 0.35
            total = 0
            sim = []
            for i in embeddings:
                cosine = cosine_similarity(q_embd, i).item()
                sim.append(cosine)
                total += cosine

            # threshold = total / len(embeddings)

            # prompt_filter = [idx for idx, p in enumerate(prompt) if sim[idx] > threshold]
            # print(f"threshold {threshold}, filtering chuck rate {len(prompt_filter) / len(prompt)}")

            # prompt = [prompt[i] for i in prompt_filter]
            # embeddings = [embeddings[i] for i in prompt_filter]

            for idx, p in enumerate(prompt):
                if sim[idx] < ALL_THRESHOLD: # 가장 스코어가 높은 답안
                    prompt[idx] = "작업전 안전교육 강화 및 작업장 위험요소 점검을 통한 재발 방지와 안전관리 교육 철저를 통한 향후 조치 계획."
            
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

            #print(f"SCD 결과 : {best_answer}")
            retriver.append(best_answer)

        if mode == "train":
            answer_text = self.data_df['재발방지대책 및 향후조치계획'].tolist()
            self.data = [{"retriver": r, "question": q, "answer": a} for r, q, a in zip(retriver, question, answer_text)]
        
        elif mode == "test" or mode == "SCD":
            self.data = [{"retriver": r, "question": q, "answer": ""} for r, q in zip(retriver, question)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def to_huggingface_dataset(self):
        return HFDataset.from_list(self.data)

class CustomDataset(Dataset):
    """
    예시 1) 한 행(row)에 대해 5개의 prompt/question을 '모두' 생성하여,
    row마다 5개씩 => 전체 row 수 * 5가 최종 데이터 크기가 됨.
    """
    def __init__(self, config: Dict[str, Any], mode: str):
        self.data_config = config
        self.data_dir = config['dir']
        self.mode = mode

        # 데이터 경로 설정
        if mode == "train":
            data_path = self.data_dir + config['train_csv']
        elif mode == "test" or mode == "SCD":
            data_path = self.data_dir + config['test_csv']
        else:
            raise ValueError("mode는 'train' 또는 'test'만 가능합니다.")

        # CSV 파일 로드
        self.data_df = pd.read_csv(data_path, encoding='utf-8-sig')

        # 컬럼 분리 (안전한 방식으로 적용)
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

        # ------------------------------
        # (예시 1) 한 행에서 5개 프롬프트 모두 만들기
        # ------------------------------
        def make_prompts_for_row(row):
            """
            한 행(row)에 대해 5가지 서로 다른 문장 템플릿을 모두 생성하여
            [ {question, answer}, {question, answer}, ... ] 리스트로 반환합니다.
            """
            # question = (
            #         # (f"{row['장소(대)']} {row['장소(소)']}에서 " if row['장소(소)'] else f"{row['장소(대)']}에서 ")
            #         #+ (f"{row['부위(대)']} {row['부위(소)']}에 " if row['부위'] != '/' else "")                
            #         f"{row['인적사고']} 인적사고"
            #         + (f", {row['물적사고']} 물적사고 발생. " if row['물적사고'] != '없음' else " 발생. ")
            #         + f"사고객체는 {row['사고객체(대분류)']}(중분류: {row['사고객체(중분류)']})로 "
            #         + f"{row['작업프로세스']}중, 사고 원인은 {row['사고원인']}일 때 "
            #         + "재발 방지 및 조치 계획은?"
            #  )
            # 템플릿 1
            question1 = (
                f"{row['인적사고']} 인적사고"
                + (f", {row['물적사고']} 물적사고가" if row['물적사고'] != '없음' else "가")
                + " 발생했습니다. "
                + f"사고객체는 {row['사고객체(대분류)']} (중분류: {row['사고객체(중분류)']}) "
                + f"이며, {row['작업프로세스']} 중에 발생한 사고의 원인은 "
                + f"{row['사고원인']}입니다. 이때, 재발 방지 및 조치 계획을 어떻게 세울까요?"
            )

            # 템플릿 2
            question2 = (
                f"다음 상황을 가정해 봅시다. "
                + f"작업프로세스 '{row['작업프로세스']}' 도중에 "
                + f"[인적사고: {row['인적사고']} / 물적사고: {row['물적사고']}]가 보고되었습니다. "
                + f"사고객체는 {row['사고객체(대분류)']} (중분류: {row['사고객체(중분류)']})이고 "
                + f"사고 원인은 {row['사고원인']}입니다. "
                + "이러한 사고에 대해 재발을 막고 피해를 최소화하기 위한 방안은 무엇일까요?"
            )

            # 템플릿 3
            question3 = (
                f"작업 현장에서 {row['인적사고']} 인적사고"
                + (f"와 {row['물적사고']} 물적사고가 동시에" if row['물적사고'] != '없음' else "가")
                + " 발생했습니다. "
                + f"사고객체는 {row['사고객체(대분류)']} (중분류: {row['사고객체(중분류)']})이고, "
                + f"사고 원인은 {row['사고원인']}로 파악되었습니다. "
                + f"{row['작업프로세스']} 상황에서 이러한 사고가 발생하지 않도록 하려면 "
                + "어떤 대책과 조치를 마련해야 할까요?"
            )

            # 템플릿 4
            question4 = (
                f"{row['사고원인']} 때문에 "
                + f"{row['인적사고']} 인적사고와 {row['물적사고']} 물적사고가 발생한 사례가 있습니다. "
                + f"당시 작업프로세스는 '{row['작업프로세스']}'였고, "
                + f"사고객체는 {row['사고객체(대분류)']} (중분류: {row['사고객체(중분류)']})였습니다. "
                + "이와 같은 유형의 사고를 예방하기 위해서는 어떤 조치와 방안이 필요할까요?"
            )

            # 템플릿 5
            question5 = (
                f"{row['인적사고']}이(가) 발생하였고 "
                + (f"{row['물적사고']}도 함께 발생한 " if row['물적사고'] != '없음' else "")
                + f"사건입니다. 사고 객체는 {row['사고객체(대분류)']}({row['사고객체(중분류)']})이며 "
                + f"작업프로세스 '{row['작업프로세스']}' 중 발생한 사고의 원인은 "
                + f"'{row['사고원인']}'입니다. "
                + "이에 대한 재발 방지책과 향후 조치 계획을 상세히 제시해 주세요."
            )

            # 모든 질문에 대한 공통 정답(Train 모드일 때만)
            if self.mode == "train":
                answer_text = str(row.get('재발방지대책 및 향후조치계획', ''))
            else:
                answer_text = ""

            # 5개의 question-answer 쌍을 리스트로 묶어서 반환
            return [
                {"question": question1, "answer": answer_text},
                {"question": question2, "answer": answer_text},
                {"question": question3, "answer": answer_text},
                {"question": question4, "answer": answer_text},
                {"question": question5, "answer": answer_text},
            ]

        # apply() → 각 행마다 5개 prompt 리스트 → 2차원 리스트가 만들어짐
        all_rows_2d = self.data_df.apply(make_prompts_for_row, axis=1).tolist()

        # Flatten 처리
        if self.mode == "train" or mode == "SCD":
            flatten_data = []
            for row_prompts in all_rows_2d:
                flatten_data.extend(row_prompts)

            # 최종적으로 self.data에 할당
            self.data = flatten_data
        else : # test 일때는 1개만 처리 
            self.data = [random.choice(row_prompts) for row_prompts in all_rows_2d]

        if self.mode == 'train':
            final_data = []
            embedding_model_name = "jhgan/ko-sbert-sts"
            model = SentenceTransformer(embedding_model_name)
            THRESHOLD = 0.405

            def cosine_similarity(a, b):
                dot_product = np.dot(a, b)
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                return dot_product / (norm_a * norm_b) if (norm_a != 0 and norm_b != 0) else 0
            
            def jaccard_similarity(text1, text2):
                """자카드 유사도 계산"""
                set1, set2 = set(text1.split()), set(text2.split())  # 단어 집합 생성
                intersection = len(set1.intersection(set2))  # 교집합 크기
                union = len(set1.union(set2))  # 합집합 크기
                return intersection / union if union != 0 else 0
            
            for d in tqdm(self.data, total=len(self.data), desc="train filtering"):
                answer_emb = model.encode(d['answer'])

                prompt_emb = model.encode(d['question'])
                
                sim = 0.7* cosine_similarity(prompt_emb, answer_emb) + 0.3*jaccard_similarity(d['answer'], d['question'])

                # threshold 넘는지 확인
                if sim >= THRESHOLD:
                    final_data.append(d)
            
            print(f"filter train len : {len(final_data)}")
            self.data = final_data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def to_huggingface_dataset(self):
        return HFDataset.from_list(self.data)

class ragdataset(Dataset):
    def __init__(self, config: Dict[str, Any], mode: str):
        self.data_config = config
        self.data_dir = config['dir']
        self.mode = mode

        # 데이터 경로 설정
        if mode == "train":
            data_path = self.data_dir + config['train_csv']
        elif mode == "test" or mode == "SCD":
            data_path = self.data_dir + config['test_csv']
        else:
            raise ValueError("mode는 'train' 또는 'test'만 가능합니다.")

        # CSV 파일 로드
        self.data_df = pd.read_csv(data_path, encoding='utf-8-sig')

        # 컬럼 분리 (안전한 방식으로 적용)
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

        rag_class = rag(self.data_config)

        # 질문 생성
        def format_prompt(row):
            # 인적사고, 공종, 사고객체, 사고 원인
            rag_prompt = (
                f"{row['인적사고']} "
                + f"{row['공종(중분류)']} "
                + f"{row['사고객체(중분류)']} "
                + f"{row['작업프로세스']} "
                + f"{row['사고원인']} "
            )
            retriver_document = rag_class.get_answer(rag_prompt)

            # retriver = ""
            # for i in retriver_document:
            #     retriver += f"{i.page_content}\n"

            
            # question1_docs = random.sample(retriver_document, 3)
            # question2_docs = random.sample(retriver_document, 3)
            # question3_docs = random.sample(retriver_document, 3)
            # question4_docs = random.sample(retriver_document, 3)
            # question5_docs = random.sample(retriver_document, 3)

            # retriver1_text = "\n".join([doc.page_content for doc in question1_docs])
            # retriver2_text = "\n".join([doc.page_content for doc in question2_docs])
            # retriver3_text = "\n".join([doc.page_content for doc in question3_docs])
            # retriver4_text = "\n".join([doc.page_content for doc in question4_docs])
            # retriver5_text = "\n".join([doc.page_content for doc in question5_docs])

            retriver_text = "\n".join([doc.page_content for doc in retriver_document])

            question1 = (
                f"{row['인적사고']} 인적사고가"
                + " 발생했습니다. "
                + f"{row['공종(중분류)']} 공종에서 사고객체는 {row['사고객체(중분류)']}"
                + f"이며, {row['작업프로세스']} 중에 발생한 사고의 원인은 "
                + f"{row['사고원인']}입니다. 이때, 재발 방지 및 조치 계획을 어떻게 세울까요?"
            )

            question2 = (
                f"{row['공종(중분류)']} 공종에서"
                + f"{row['사고객체(중분류)']}에 의해 {row['인적사고']} 인적사고가 발생했습니다."
                + f"사고 원인이 {row['사고원인']}일 때, 안전을 지킬 수 있는 방법이 무엇인가요?"
            )

            question3 = (
                f"{row['사고객체(중분류)']}를 통해 {row['작업프로세스']} 중에"
                + f" {row['인적사고']} 사고를 당했습니다."
                + f"원인은 {row['사고원인']}일 때, 재발 방지 및 조치 계획은 무엇인가요?"
            )

            if self.mode == "train":
                answer = f"{row['재발방지대책 및 향후조치계획']}"
            else:  # test 모드에서는 정답 없음
                answer = ""

            return [
                {"retriver": retriver_text, "question": question1, "answer": answer},
                {"retriver": retriver_text, "question": question2, "answer": answer},
                {"retriver": retriver_text, "question": question3, "answer": answer}
            ]

        all_rows_2d = self.data_df.apply(format_prompt, axis=1).tolist()
        
        # Flatten 처리
        if self.mode == "train" or mode == "SCD":
            flatten_data = []
            for row_prompts in all_rows_2d:
                flatten_data.extend(row_prompts)

            # 최종적으로 self.data에 할당
            self.data = flatten_data
        else : # test 일때는 1개만 처리 
            self.data = [random.choice(row_prompts) for row_prompts in all_rows_2d]

        if self.mode == 'train':
            final_data = []
            embedding_model_name = "jhgan/ko-sbert-sts"
            model = SentenceTransformer(embedding_model_name)
            THRESHOLD = 0.405

            def cosine_similarity(a, b):
                dot_product = np.dot(a, b)
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                return dot_product / (norm_a * norm_b) if (norm_a != 0 and norm_b != 0) else 0
            
            for d in tqdm(self.data, total=len(self.data), desc="train filtering"):
                answer_emb = model.encode(d['answer'])

                prompt_emb = model.encode(d['question'])
                sim = cosine_similarity(prompt_emb, answer_emb)

                # threshold 넘는지 확인
                if sim >= THRESHOLD:
                    final_data.append(d)
            
            print(f"filter train len : {len(final_data)}")
            self.data = final_data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def to_huggingface_dataset(self):
        return HFDataset.from_list(self.data)



if __name__ == '__main__':
    def get_config(config_folder):
        config = {}

        config_folder = os.path.join(config_folder,'*.yaml')
        config_files = glob.glob(config_folder)
        for file in config_files:
            with open(file, 'r') as f:
                config.update(yaml.safe_load(f))

        if config.get('device', 'cpu') == 'cuda' and not torch.cuda.is_available():
            print('using cpu now...')
            config['device'] = 'cpu'

        return config

    config = get_config("/home/aicontest/construct/experiment/configs")

    dataset = CustomDataset(config['data'], mode='train')

    for i in dataset[:10]:
        print(i)
