#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# app.py
from fastapi import FastAPI, Query
from core import search_related

# FastAPIアプリを作成
app = FastAPI(
    title="Incident Search API",
    description="ICOS上の障害記録をベクトル検索するAPI",
    version="1.0"
)

# エンドポイント定義
@app.get("/search")
def search(q: str = Query(..., description="検索クエリ（質問文など）")):
    """
    クエリ文字列 q を受け取り、類似する障害記録を返す。
    """
    results = search_related(q)
    return {"query": q, "results": results}

# ルート確認用
@app.get("/")
def root():
    return {"message": "API is running. Try /search?q=サーバーが落ちた"}

