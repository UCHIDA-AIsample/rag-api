#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# core.py
import os
import io
import re
import pandas as pd
import ibm_boto3
from ibm_botocore.client import Config
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# --- テキスト整形関数 ---
def clean_past_records(raw_text: str) -> str:
    """
    過去障害記録のテキストを整形してクリーンなフォーマットにする
    - 行頭・行末の空白削除
    - タブをスペースに変換
    - 複数スペースを1つに変換
    - 空行は削除
    """
    cleaned_lines = []
    for line in raw_text.splitlines():
        line = line.replace("\t", " ")      # タブをスペースに変換
        line = line.strip()                 # 行頭・行末の空白削除
        line = re.sub(r"\s{2,}", " ", line) # 複数スペースを1つに
        if line:  # 空行はスキップ
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

# --- ICOS 接続設定（環境変数から読み込む） ---
cos_api_key = os.environ["IBM_API_KEY"]
cos_resource_instance_id = os.environ["RESOURCE_CRN"]
cos_endpoint = os.environ["ENDPOINT"]
bucket_name = os.environ["BUCKET"]
object_key = os.environ["CSV_KEY"]

cos = ibm_boto3.client(
    service_name="s3",
    ibm_api_key_id=cos_api_key,
    ibm_service_instance_id=cos_resource_instance_id,
    config=Config(signature_version="oauth"),
    endpoint_url=cos_endpoint,
)

# --- 1. COSからCSV読み込み ---
response = cos.get_object(Bucket=bucket_name, Key=object_key)
df = pd.read_csv(io.BytesIO(response["Body"].read()))

# --- 2. ドキュメント作成 ---
search_texts = []
full_payloads = []

for idx, row in df.iterrows():
    # 検索用: 内容
    search_text = str(row["障害内容"])
    search_texts.append(search_text)

    # 表示用: 原因と対処法
    full_text_raw = f"""
    原因: {row['障害原因']}
    対処法: {row['暫定対応内容']}
    """
    full_text_clean = clean_past_records(full_text_raw)
    full_payloads.append(full_text_clean)

# --- 3. Embeddingモデル（日本語対応） ---
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(search_texts)

# --- 4. Qdrant（インメモリ）でベクトルDB作成 ---
qdrant = QdrantClient(":memory:")
qdrant.recreate_collection(
    collection_name="shogai",
    vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE),
)

for i, vector in enumerate(embeddings):
    qdrant.upsert(
        collection_name="shogai",
        points=[PointStruct(id=i, vector=vector, payload={"text": full_payloads[i]})],
    )

# --- 5. 検索関数（APIから呼び出す用） ---
def search_related(query: str, top_k: int = 3, min_score: float = 0.6):
    """
    クエリ（質問）を受け取り、類似度の高い障害記録を返す
    """
    query_emb = model.encode([query])[0]
    hits = qdrant.search(collection_name="shogai", query_vector=query_emb, limit=top_k)

    # 類似度スコアが min_score 以上のものだけ返す
    filtered_hits = [
        {"score": hit.score, "text": hit.payload["text"]}
        for hit in hits
        if hit.score >= min_score
    ]
    return filtered_hits

