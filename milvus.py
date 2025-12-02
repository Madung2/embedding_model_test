from pymilvus import (
    MilvusClient, DataType, Function, FunctionType
)
import os
from dotenv import load_dotenv
load_dotenv()
MILVUS_URL = os.getenv("MILVUS_URL", "http://localhost:19530")

# Create client
client = MilvusClient(uri=MILVUS_URL, token="root:Milvus")

COLLECTION = "cmd_history_minilm"

# === 2. 기존 컬렉션 있으면 삭제(Optional) ===
if client.has_collection(COLLECTION):
    client.drop_collection(COLLECTION)

# === 3. 스키마 생성 ===
schema = client.create_schema(
    auto_id=True  # 내부적으로 PK 자동 생성
)

# 텍스트 저장 (검색용 X, metadata 용)
schema.add_field(
    field_name="text",
    datatype=DataType.VARCHAR,
    max_length=1000,
    description="cleaned raw command line"
)

# 원본 라인 번호
schema.add_field(
    field_name="line",
    datatype=DataType.INT64,
    description="line number"
)

# MiniLM 임베딩 384차원
schema.add_field(
    field_name="vector",
    datatype=DataType.FLOAT_VECTOR,
    dim=384,
    description="MiniLM embedding vector (384-dim)"
)

# === 4. 컬렉션 생성 ===
client.create_collection(
    collection_name=COLLECTION,
    schema=schema,
    shards_num=2
)

print(f"Created collection: {COLLECTION}")

# === 5. 벡터 인덱스 생성 ===
client.create_index(
    collection_name=COLLECTION,
    index_name="idx_vector",
    field_name="vector",
    index_params={
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "nlist": 1024
    }
)

print("Index created.")