import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from app.rag_chain import RAGChain

st.set_page_config(
    page_title="RAGEval Dashboard",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 RAGEval — RAG Evaluation Dashboard")
st.markdown("Evaluating hallucination rate, retrieval quality, and groundedness across 15 AI research papers.")

# ── Load evaluation results ──
@st.cache_data
def load_results():
    return pd.read_csv("results/eval_results.csv")

df = load_results()

# ── Top KPI metrics ──
st.markdown("## 📊 Evaluation Summary")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Avg Groundedness", f"{df['groundedness_score'].mean():.2f}")
with col2:
    hallucination_rate = (df['groundedness_score'] < 0.5).mean() * 100
    st.metric("Hallucination Rate", f"{hallucination_rate:.1f}%")
with col3:
    st.metric("Avg Retrieval Score", f"{df['retrieval_score'].mean():.3f}")
with col4:
    st.metric("Questions Evaluated", len(df))

st.divider()

# ── Groundedness distribution ──
st.markdown("## 🧠 Groundedness Score Distribution")
fig1 = px.histogram(
    df, x="groundedness_score",
    nbins=10,
    color_discrete_sequence=["#4CAF50"],
    title="Distribution of Groundedness Scores"
)
fig1.update_layout(xaxis_title="Groundedness Score", yaxis_title="Count")
st.plotly_chart(fig1, use_container_width=True)

st.divider()

# ── Retrieval vs Groundedness scatter ──
st.markdown("## 📈 Retrieval Quality vs Groundedness")
fig2 = px.scatter(
    df,
    x="retrieval_score",
    y="groundedness_score",
    color="groundedness_score",
    color_continuous_scale="RdYlGn",
    title="Retrieval Score vs Groundedness Score",
    hover_data=["question"]
)
fig2.update_layout(xaxis_title="Retrieval Score", yaxis_title="Groundedness Score")
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Sample results table ──
st.markdown("## 📋 Sample Evaluation Results")
st.dataframe(
    df[["question", "generated_answer", "groundedness_score", "retrieval_score", "reasoning"]].head(20),
    use_container_width=True
)

st.divider()

# ── Live RAG Query ──
st.markdown("## 🚀 Live RAG Query")
st.markdown("Ask a question about the 15 research papers:")

@st.cache_resource
def load_rag():
    return RAGChain()

question = st.text_input("Enter your question:", placeholder="What is the attention mechanism?")

if question:
    with st.spinner("Retrieving and generating answer..."):
        rag = load_rag()
        result = rag.query(question)
    
    st.markdown("### Answer")
    st.write(result["answer"])
    
    st.markdown("### Retrieved Context")
    for i, doc in enumerate(result["context_docs"]):
        with st.expander(f"Chunk {i+1} — {doc.metadata.get('source', 'unknown')}"):
            st.write(doc.page_content)