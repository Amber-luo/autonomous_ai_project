from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path for import src
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.llm import qna_ai
from src.utils.storage import read_parquet


REPORTS_DIR = Path("reports")
PROCESSED_DIR = Path("data/processed")


def _list_processed_files() -> list[Path]:
    if not PROCESSED_DIR.exists():
        return []
    return sorted(PROCESSED_DIR.glob("*_dataset.parquet"))


def _find_report_for(processed_path: Path) -> Path | None:
    report_name = processed_path.stem + "_report.txt"
    report_path = REPORTS_DIR / report_name
    return report_path if report_path.exists() else None


def _build_context(df) -> str:
    parts = []
    parts.append(f"rows={len(df)}")
    parts.append(f"columns={list(df.columns)}")
    if "anomaly" in df.columns:
        total = len(df)
        anomaly_count = int((df["anomaly"] == -1).sum())
        ratio = anomaly_count / total if total else 0
        parts.append(f"anomaly_count={anomaly_count}")
        parts.append(f"anomaly_ratio={ratio:.3f}")
    return "\n".join(parts)


def _save_uploaded_parquet(uploaded_file) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / uploaded_file.name
    out_path.write_bytes(uploaded_file.getbuffer())
    return out_path


def launch_dashboard() -> None:
    st.set_page_config(page_title="Autonomous AI Platform", layout="wide")
    st.title("Autonomous AI Platform Dashboard")

    # ------------------------------------------------------
    # 0) 上传本地真实采集后的 processed 文件
    # ------------------------------------------------------
    st.subheader("Upload Processed Dataset (from local CARLA run)")
    uploaded = st.file_uploader(
        "Upload a processed parquet file (*_dataset.parquet)",
        type=["parquet"],
    )
    if uploaded is not None:
        saved_path = _save_uploaded_parquet(uploaded)
        st.success(f"Uploaded: {saved_path.name}")

    # ------------------------------------------------------
    # 1) 选择 processed 数据文件
    # ------------------------------------------------------
    files = _list_processed_files()
    if not files:
        st.info("No processed dataset found. Upload one first.")
        return

    file_labels = [f.name for f in files]
    selected = st.selectbox("Select processed dataset", file_labels)
    processed_path = PROCESSED_DIR / selected

    # ------------------------------------------------------
    # 2) 展示数据预览
    # ------------------------------------------------------
    df = read_parquet(processed_path)
    st.subheader("Processed Dataset Preview")
    st.write(df.head(20))

    # ------------------------------------------------------
    # 3) 异常统计
    # ------------------------------------------------------
    st.subheader("Anomaly Summary")
    if "anomaly" in df.columns:
        total = len(df)
        anomaly_count = int((df["anomaly"] == -1).sum())
        ratio = anomaly_count / total if total else 0
        st.write({"total": total, "anomaly_count": anomaly_count, "anomaly_ratio": ratio})
        st.bar_chart(df["anomaly"].value_counts())
    else:
        st.info("No anomaly column found in this dataset.")

    # ------------------------------------------------------
    # 4) 报告展示
    # ------------------------------------------------------
    st.subheader("Latest Report")
    report_path = _find_report_for(processed_path)
    if report_path is not None:
        st.code(report_path.read_text(encoding="utf-8"))
    else:
        st.info("No report found for this dataset. Run training/report step first.")

    # ------------------------------------------------------
    # 5) 自然语言问答（Ollama + RAG）
    # ------------------------------------------------------
    st.subheader("LLM Query (Ollama + RAG)")
    use_rag = st.checkbox("Enable RAG", value=True)
    model_name = st.text_input("Model name", value="qwen2.5")
    query = st.text_input("Ask a question")

    if query:
        if use_rag:
            answer, hits = qna_ai.query_rag(query, llm_model=model_name, top_k=3)
            st.write(answer)
            if hits:
                with st.expander("RAG Retrieved Sources"):
                    for h in hits:
                        st.write({"source": h["source"], "score": h["score"]})
                        st.code(h["text"])
        else:
            context = _build_context(df)
            st.write(qna_ai.query(query, llm_model=model_name, context=context))


if __name__ == "__main__":
    launch_dashboard()
