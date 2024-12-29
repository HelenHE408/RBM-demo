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

    def ask_glm(self, content):
        url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': self.generate_token("41a0f931aeba1ea2ceeac89afbcd95ef.mQSOD4dJAhcL6yJh", 1000)
        }

        data = {
            "model": "glm-4",
            "messages": [{"role": "user", "content": content}]
        }

        response = requests.post(url, headers=headers, json=data)
        return response.json()

    def get_glm(self, content):
        client = ZhipuAI(api_key="bfb70f16f00136c77adbe41c44cbca45.8svJMtDNjEOy9Xjy")
        response = client.chat.completions.create(
            model="glm-4",  # 请填写您要调用的模型名称
            messages=[
                {"role": "user",
                 "content": content},
            ],
            stream=True,
        )
        # for chunk in response:
        #     print(chunk.choices[0].delta)
        return response

    def ask_Qwen(self, content):
        responses = dashscope.Generation.call(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key='sk-51d2f716cbdd4be08235def036f8e735',
            model="qwen1.5-1.8b-chat",
            messages=[{'role': 'user', 'content': content}],
            result_format='message',
            stream=True,
            incremental_output=True,
            temperature=0.2,
        )
        # for response in responses:
        #     print(response["output"]["choices"][0]["message"]["content"])

        return responses
        # return response.output.choices[0].message.content


    def generate_token(self, apikey: str, exp_seconds: int):
        try:
            id, secret = apikey.split(".")
        except Exception as e:
            raise Exception("invalid apikey", e)

        payload = {
            "api_key": id,
            "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
            "timestamp": int(round(time.time() * 1000)),
        }
        return jwt.encode(
            payload,
            secret,
            algorithm="HS256",
            headers={"alg": "HS256", "sign_type": "SIGN"},
        )

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
            print('\n\n\n')
            print(material)
            # os.system('clear')
        else:
            responses = self.get_glm(prompt)
            print('\n\n\n')
            for response in responses:
                print(response.choices[0].delta.content, end='')

    def ask_question(self):
        question = input("请输入你的问题 (输入 'exit' 退出): ")
        if question.lower() == 'exit':
            print("程序结束。")
            exit()  # 退出程序
        elif len(question)>1:
            self.add_question(question)
            self.answer_questions()
            print('\n\n\n')
        # answer = self.questions[-1]['answer']
            self.questions.pop()
        else:
            print('empty question')


if __name__ == "__main__":
    text_qa = TextQA("data/corpus.json")

    while True:
        text_qa.ask_question()