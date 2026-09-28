import streamlit as st
import torch

from medguiderag.config import EMBEDDING_MODEL_CANDIDATES
from medguiderag.generation.generator import generate_answer
from medguiderag.retrieval.hybrid import HybridRetriever
from medguiderag.retrieval.retriever import Retriever

st.set_page_config(page_title="MedGuideRag", page_icon="💊")
st.title("MedGuideRag")
st.caption(
    "Демонстрационен прототип: въпроси към национални фармакотерапевтични "
    "ръководства чрез локална система за извличане с генериране."
)


@st.cache_resource
def get_retriever(model_name: str, hybrid: bool) -> Retriever | HybridRetriever:
    return HybridRetriever(model_name) if hybrid else Retriever(model_name)


with st.sidebar:
    embedding_model = st.selectbox("Модел за векторно представяне", EMBEDDING_MODEL_CANDIDATES)
    # The reranker takes ~1 s per question on the GPU but ~70 s on the CPU.
    hybrid = st.toggle(
        "Хибридно извличане с пренареждане",
        value=torch.cuda.is_available(),
        help="Кандидатите от векторното търсене и от BM25 се обединяват и се подреждат наново "
        "от модел за пренареждане; откъси с ниска оценка на релевантност се отхвърлят.",
    )
    # ~1100 prompt tokens per chunk: more than 5 no longer fits the model's 8192 context with the answer.
    top_k = st.slider("Брой извлечени откъси (k)", min_value=1, max_value=5, value=5)

question = st.text_input("Задайте медицински въпрос:")

if question:
    retriever = get_retriever(embedding_model, hybrid)
    with st.spinner("Извличане на релевантни откъси..."):
        chunks = retriever.search(question, k=top_k)

    if not chunks:
        # Every candidate scored below RERANK_MIN_SCORE: the guidelines most likely do not cover the question.
        st.warning(
            "В ръководствата не бяха открити откъси, свързани с въпроса, затова отговор не беше генериран."
        )
        st.stop()

    with st.spinner("Генериране на отговор..."):
        answer = generate_answer(question, chunks)

    st.subheader("Отговор")
    st.write(answer)
    st.caption(
        "Отговорът е генериран от езиков модел въз основа на откъсите по-долу и може да съдържа "
        "неточности. Проверявайте информацията в посочените източници."
    )

    st.subheader("Източници")
    score_label = "релевантност" if hybrid else "прилика"
    for c in chunks:
        with st.expander(f"{c['document']} · стр. {c['page_start']}-{c['page_end']} · {score_label} {c['score']:.3f}"):
            st.caption(c["section"])
            st.write(" ".join(c["text"].split()))
