from __future__ import annotations

"""
QnA (Ollama)
- 默认通过本地 Ollama API 调用开源模型
- 可选 RAG：先检索，再生成
"""

import os
from typing import Any, Optional

import requests

from src.rag.index import build_index, retrieve

DEFAULT_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
DEFAULT_MODEL = os.environ.get('OLLAMA_MODEL', 'qwen2.5')


def _call_ollama(prompt: str, model: str) -> str:
    payload = {
        'model': model,
        'prompt': prompt,
        'stream': False,
    }
    resp = requests.post(f"{DEFAULT_BASE_URL}/api/generate", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data.get('response', '').strip() or '模型没有返回内容。'


def query(
    prompt: str,
    llm_model: Any | None = None,
    context: Optional[str] = None,
) -> str:
    """
    普通问答（不使用 RAG）
    """
    model = llm_model or DEFAULT_MODEL
    system = (
        '你是自动驾驶数据平台的助手。'
        '请基于给定上下文回答问题，如果上下文不足就说明需要更多数据。'
    )
    if context:
        user = f"上下文：\n{context}\n\n问题：{prompt}"
    else:
        user = prompt
    return _call_ollama(f"{system}\n\n{user}", model)


def query_rag(prompt: str, llm_model: Any | None = None, top_k: int = 3) -> tuple[str, list[dict]]:
    """
    RAG 问答：
    1) 构建索引
    2) 检索 top_k 文档
    3) 拼接上下文 → 生成
    返回：回答 + 检索结果
    """
    model = llm_model or DEFAULT_MODEL
    index = build_index()
    hits = retrieve(prompt, index, top_k=top_k)

    if hits:
        context = "\n\n".join([f"[source] {h['source']}\n{h['text']}" for h in hits])
    else:
        context = "(无可检索文档)"

    system = (
        '你是自动驾驶数据平台的助手。'
        '请根据检索到的资料回答问题，并尽量引用来源。'
        '如果资料不足，请明确说明。'
    )
    user = f"检索到的资料：\n{context}\n\n问题：{prompt}"
    answer = _call_ollama(f"{system}\n\n{user}", model)
    return answer, hits
