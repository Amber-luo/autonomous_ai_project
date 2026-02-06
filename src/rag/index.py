from __future__ import annotations

"""
RAG 简易检索模块（TF-IDF 版）
- 不依赖大型向量库，适合快速跑通
- 后续可替换为 FAISS/Chroma + Embedding
"""

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RagDoc:
    source: str
    text: str


@dataclass
class RagIndex:
    vectorizer: TfidfVectorizer
    matrix
    docs: list[RagDoc]


def _load_reports(reports_dir: Path) -> list[RagDoc]:
    docs: list[RagDoc] = []
    if not reports_dir.exists():
        return docs
    for path in sorted(reports_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        docs.append(RagDoc(source=str(path), text=text))
    return docs


def _load_meta(raw_dir: Path) -> list[RagDoc]:
    docs: list[RagDoc] = []
    if not raw_dir.exists():
        return docs
    for path in sorted(raw_dir.glob("*/meta.json")):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        text = json.dumps(meta, ensure_ascii=False, indent=2)
        docs.append(RagDoc(source=str(path), text=text))
    return docs


def _summarize_processed(df: pd.DataFrame, source: str) -> str:
    lines = []
    lines.append(f"source={source}")
    lines.append(f"rows={len(df)}")
    lines.append(f"columns={list(df.columns)}")
    if "anomaly" in df.columns:
        total = len(df)
        anomaly_count = int((df["anomaly"] == -1).sum())
        ratio = anomaly_count / total if total else 0
        lines.append(f"anomaly_count={anomaly_count}")
        lines.append(f"anomaly_ratio={ratio:.3f}")
    # 只取少量样例，避免过长
    sample = df.head(3).to_dict(orient="records")
    lines.append(f"sample={sample}")
    return "\n".join(lines)


def _load_processed(processed_dir: Path) -> list[RagDoc]:
    docs: list[RagDoc] = []
    if not processed_dir.exists():
        return docs
    for path in sorted(processed_dir.glob("*_dataset.parquet")):
        df = pd.read_parquet(path)
        text = _summarize_processed(df, source=str(path))
        docs.append(RagDoc(source=str(path), text=text))
    return docs


def build_index(
    processed_dir: str | Path = "data/processed",
    reports_dir: str | Path = "reports",
    raw_dir: str | Path = "data/raw",
) -> RagIndex:
    processed_dir = Path(processed_dir)
    reports_dir = Path(reports_dir)
    raw_dir = Path(raw_dir)

    docs = []
    docs.extend(_load_processed(processed_dir))
    docs.extend(_load_reports(reports_dir))
    docs.extend(_load_meta(raw_dir))

    corpus = [d.text for d in docs]
    if not corpus:
        # 空索引
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform([""])
        return RagIndex(vectorizer=vectorizer, matrix=matrix, docs=[])

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    return RagIndex(vectorizer=vectorizer, matrix=matrix, docs=docs)


def retrieve(query: str, index: RagIndex, top_k: int = 3) -> list[dict]:
    if not index.docs:
        return []
    q_vec = index.vectorizer.transform([query])
    sims = cosine_similarity(q_vec, index.matrix)[0]
    ranked = sims.argsort()[::-1][:top_k]
    results = []
    for i in ranked:
        doc = index.docs[int(i)]
        results.append({"source": doc.source, "text": doc.text, "score": float(sims[int(i)])})
    return results
