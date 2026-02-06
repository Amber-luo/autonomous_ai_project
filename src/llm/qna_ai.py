from __future__ import annotations

"""
QnA (Ollama)
- 默认通过本地 Ollama API 调用开源模型
- 你可以设置环境变量 OLLAMA_BASE_URL 或 OLLAMA_MODEL
"""

import os
from typing import Any, Optional

import requests

DEFAULT_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
DEFAULT_MODEL = os.environ.get('OLLAMA_MODEL', 'qwen2.5')


def query(prompt: str, llm_model: Any | None = None, context: Optional[str] = None) -> str:
    """
    用 Ollama 进行本地推理。
    - prompt: 用户问题
    - context: 可选上下文（来自数据摘要）
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

    payload = {
        'model': model,
        'prompt': user,
        'system': system,
        'stream': False,
    }

    try:
        resp = requests.post(f"{DEFAULT_BASE_URL}/api/generate", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get('response', '').strip() or '模型没有返回内容。'
    except requests.exceptions.RequestException as exc:
        return f"调用 Ollama 失败：{exc}"
