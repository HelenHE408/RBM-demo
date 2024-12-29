import subprocess
import os
import copy
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagAutoModel
import time
import jwt
import requests
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from rank_bm25 import BM25Okapi
from zhipuai import ZhipuAI

import os
from dashscope import Generation
import dashscope

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import warnings

# 禁用所有 UserWarning 类型的警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="transformers.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TextQA:
    def __init__(self,
                 txt_file_path,
                 ):

        self.txt_file_path = txt_file_path
        self.chunk_content = []
        self.questions = []
        self._load_environment()
        self._load_text_data()
        self._split_text()

    def _load_environment(self):
        result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True,
                                capture_output=True, text=True)
        output = result.stdout
        for line in output.splitlines():
            if '=' in line:
                var, value = line.split('=', 1)
                os.environ[var] = value

    def _load_text_data(self):
        # with open(self.txt_file_path, 'r', encoding='utf-8') as file:
        #     self.txt_content = file.read()
        with open(self.txt_file_path, 'r', encoding='utf-8') as file:
            self.txt_content = json.load(file)

    def _split_text(self):
        txt_lines = []
        for qa in self.txt_content:
            txt_lines.append(qa['question'])
        for idx, line in enumerate(self.txt_content):
            self.chunk_content.append({
                'answer': self.txt_content[idx]['answer'],
                'content': self.txt_content[idx]['question'],
                'chunk': f'chunk_{idx}',
            })

    def add_question(self, question):
        self.questions.append({'question': question, 'answer': None})

    def _tfidf_retrieval(self):
        question_words = [' '.join(jieba.lcut(q['question'])) for q in self.questions]
        content_words = [' '.join(jieba.lcut(c['content'])) for c in self.chunk_content]

        tfidf = TfidfVectorizer()
        tfidf.fit(question_words + content_words)

        question_feat = tfidf.transform(question_words)
        content_feat = tfidf.transform(content_words)

        question_feat = normalize(question_feat)
        content_feat = normalize(content_feat)

        for query_idx, feat in enumerate(question_feat):
            score = feat @ content_feat.T
            score = score.toarray().flatten()  # 转换为一维数组
            max_score_page_idx = score.argsort()[-1]  # 获取最大分数的索引
            self.questions[query_idx][f'reference'] = f'chunk_{max_score_page_idx + 1}'


    def _embedding_retrieval(self):
        model = FlagAutoModel.from_finetuned('BAAI/bge-base-en-v1.5', use_fp16=True)

        question_sentences = [q['question'] for q in self.questions]
        content_sentences = [c['content'] for c in self.chunk_content]

        question_embeddings = model.encode(question_sentences)
        chunk_embeddings = model.encode(content_sentences)

        for query_idx, feat in enumerate(question_embeddings):
            score = feat @ chunk_embeddings.T
            max_score_page_idx = score.argsort()[-1]  # 获取最大分数的索引
            self.questions[query_idx][f'reference'] = f'chunk_{max_score_page_idx + 1}'


    def _rerank_answers(self):
        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-base')
        rerank_model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-base')
        rerank_model.cuda()

        bm25 = BM25Okapi([' '.join(jieba.lcut(c['content'])) for c in self.chunk_content])

        for query_idx in range(len(self.questions)):
            doc_scores = bm25.get_scores(jieba.lcut(self.questions[query_idx]["question"]))
            max_score_page_idxs = doc_scores.argsort()[-1:]

            pairs = [[self.questions[query_idx]["question"], self.chunk_content[idx]['content']] for idx in
                     max_score_page_idxs]

            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            with torch.no_grad():
                inputs = {key: inputs[key].cuda() for key in inputs.keys()}
                scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()

            max_score_page_idx = max_score_page_idxs[scores.cpu().numpy().argmax()]
            self.questions[query_idx]['reference'] = f'chunk_{max_score_page_idx + 1}'

    def get_glm(self, content):
        client = ZhipuAI(api_key="bfb70f16f00136c77adbe41c44cbca45.8svJMtDNjEOy9Xjy")
        # client = ZhipuAI(api_key="a477c2ef9cf86f70bbb1b04ddc281a28.ZrLM5iU7JJYvIYUj")
        response = client.chat.completions.create(
            model="glm-4",  # 请填写您要调用的模型名称
            messages=[
                {"role": "user",
                 "content": content},
            ],
        )
        return response.choices[0].message.content

    def answer_questions(self):
        # self._tfidf_retrieval()
        self._embedding_retrieval()
        # self._rerank_answers()

        chunk = self.questions[-1]['reference']
        idx = int(chunk.split('_')[1]) - 1
        material = self.chunk_content[idx]['answer']

        prompt = f''' Answer the question: {self.questions[0]["question"]} 
               based on the following information {material}.
               '''
        q = list(self.questions[-1]['question'].split())
        if 'Lionel' in q or 'proof' in q:
            return material
        else:
            return self.get_glm(prompt)


    def ask_question(self):
        question = input("请输入你的问题 (输入 'exit' 退出): ")
        if question.lower() == 'exit':
            print("程序结束。")
            exit()  # 退出程序
        self.add_question(question)
        answer = self.answer_questions()
        print('\n\n\n')
        print(answer)

        self.questions.pop()


if __name__ == "__main__":
    text_qa = TextQA("data/corpus.json")

    while True:
        text_qa.ask_question()