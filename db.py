# from sentence_transformers import SentenceTransformer
# sentences = ["This is an example sentence", "Each sentence is converted"]

# model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
# embeddings = model.encode(sentences)
# print(embeddings)
from typing import Literal
from sentence_transformers import SentenceTransformer

ModelName = Literal[
    'sentence-transformers/all-MiniLM-L12-v2',
    'sentence-transformers/paraphrase-MiniLM-L6-v2',
    'sentence-transformers/msmarco-MiniLM-L-6-v3',
    'sentence-transformers/distilbert-base-nli-stsb-mean-tokens',
]



class EmbeddingModel(SentenceTransformer):

    def __init__(self, embedding_name: ModelName):
        self.embedding_name = embedding_name
        super().__init__(self.embedding_name)

    def get_model(self):
        return SentenceTransformer(self.embedding_name)



# model = EmbeddingModel('sentence-transformers/all-MiniLM-L12-v2')
# sentences = ["This is an example sentence", "Each sentence is converted"]
# embeddings = model.encode(sentences)



###### 임베딩 모델을 사용해서 한줄씩 임베딩하는 함수 만들기 ######
file = "history_cleaned.txt"


def read_line(line):
    for line in open(file, "r", encoding="utf-8"):
        print(line.strip())
        




from pydantic import BaseModel

class EmbeddingConfig(BaseModel):
    embedding_name: ModelName = 'sentence-transformers/all-MiniLM-L12-v2'
    dimension: int = 384


