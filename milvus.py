from pymilvus import (
    MilvusClient, DataType, Function, FunctionType
)
import os
from dotenv import load_dotenv
load_dotenv()


class MilvusCollectionManager:
    def __init__(self, collection_name: str, dim: int, milvus_url: str = None):
        self.collection_name = collection_name # 컬랙션 이름: 예 cmd_history_bge
        self.dim = dim

        self.milvus_url = milvus_url or os.getenv("MILVUS_URL", "http://localhost:19530")
        self.client = MilvusClient(uri=self.milvus_url, token="root:Milvus")

    def drop_if_exists(self):
        if self.client.has_collection(self.collection_name):
            print(f"[INFO] Dropping existing collection: {self.collection_name}")
            self.client.drop_collection(self.collection_name)

    def build_schema(self):
        schema = self.client.create_schema(auto_id=True)

        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=1000,
            description="cleaned raw command line",
        )

        schema.add_field(
            field_name="line",
            datatype=DataType.INT64,
            description="line number",
        )

        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.dim,
            description=f"embedding vector ({self.dim}-dim)",
        )

        schema.add_field(
            field_name="timestamp",
            datatype=DataType.INT64,
            description="timestamp",
        )

        return schema

    def create_collection(self, shards_num: int = 2):
        schema = self.build_schema()

        print(f"[INFO] Creating collection: {self.collection_name}")

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            shards_num=shards_num,
        )

    def create_index(self):
        print(f"[INFO] Creating index on `{self.collection_name}.vector`")

        self.client.create_index(
            collection_name=self.collection_name,
            index_name="idx_vector",
            field_name="vector",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "nlist": 1024,
            },
        )

    def setup(self):
        """한 번에 drop → create → index"""
        self.drop_if_exists()
        self.create_collection()
        self.create_index()
        print(f"[INFO] Setup complete for collection `{self.collection_name}`")


# ------------------------------------------------
# 사용 예시
# ------------------------------------------------

if __name__ == "__main__":
    manager = MilvusCollectionManager(
        collection_name="cmd_history_minilm",
        dim=384,
    )

    manager.setup()