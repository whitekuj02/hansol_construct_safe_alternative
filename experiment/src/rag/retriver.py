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
import re
from typing import Any, Dict

from .pdfReader import pdf_reader

class rag:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

        self.embedding_model_name = config['rag']['embedding_model']  # 임베딩 모델 선택
        self.embedding = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

        self.pdf_text = pdf_reader(config).get_text()

        self.pdf_chunk = []

        for text in self.pdf_text:
            pattern = r"\d+\.\d+|\d+\.|\(\d+\)|\([가-하]\)|○"
            
            # 정규식으로 분할하면서 구분자도 포함
            parts = re.split(pattern, text)
            
            # 빈 문자열 제거 및 양쪽 공백 제거
            parts = [p.strip() for p in parts if p.strip()]

            for i in parts:
                if i.endswith("다.") and len(i) > 30 and len(i) <= 120:
                    self.pdf_chunk.append(i)

        self.vector_store = FAISS.from_texts(self.pdf_chunk, self.embedding, distance_strategy = DistanceStrategy.COSINE)

        self.retriever = self.vector_store.as_retriever(search_type=config['rag']['search_type'], search_kwargs={**config['rag']['search_kwargs']})

    def get_answer(self, query):
        return self.retriever.get_relevant_documents(query)

