import os

# All models are already on disk, but loading one from the Hugging Face cache still asks huggingface.co for
# newer revisions. Must be set before huggingface_hub is imported; HF_HUB_OFFLINE=0 allows a first download.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import html

import streamlit as st
import torch

from medguiderag.config import EMBEDDING_MODEL_CANDIDATES, GENERATOR_MODELS
from medguiderag.generation.attribution import Attributor, normalized_text
from medguiderag.generation.coverage import CoverageChecker, refusal_message
from medguiderag.generation.generator import generate_answer
from medguiderag.generation.grounding import GroundingChecker, highlight
from medguiderag.indexing.embed import embed_texts
from medguiderag.ingest.load_chunks import load_chunks
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


@st.cache_resource
def get_checker() -> GroundingChecker:
    return GroundingChecker([c["text"] for c in load_chunks()])


@st.cache_resource
def get_coverage() -> CoverageChecker:
    return CoverageChecker([c["text"] for c in load_chunks()])


@st.cache_resource
def get_attributor(model_name: str) -> Attributor:
    # The retrieval model is already loaded, so matching sentences costs no extra memory.
    return Attributor(lambda texts: embed_texts(texts, model_name, kind="query"))


def marked(text: str, spans: list[tuple[int, int]]) -> str:
    """Chunk text as HTML with the supporting sentences in <mark>; inside a <div> Streamlit leaves it unparsed as markdown."""
    out, pos = [], 0
    for start, end in spans:
        out += [html.escape(text[pos:start]), f"<mark>{html.escape(text[start:end])}</mark>"]
        pos = end
    return f"<div>{''.join(out)}{html.escape(text[pos:])}</div>"


with st.sidebar:
    embedding_model = st.selectbox("Модел за векторно представяне", EMBEDDING_MODEL_CANDIDATES)
    # The reranker takes ~1 s per question on the GPU but ~70 s on the CPU.
    hybrid = st.toggle(
        "Хибридно извличане с пренареждане",
        value=torch.cuda.is_available(),
        help="Кандидатите от векторното търсене и от BM25 се обединяват и се подреждат наново "
        "от модел за пренареждане; откъси с ниска оценка на релевантност се отхвърлят.",
    )
    generator = st.selectbox("Генеративен модел", list(GENERATOR_MODELS), format_func=GENERATOR_MODELS.get)
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

    coverage = get_coverage().check(question, chunks)
    if not coverage.answerable:
        # Medical questions outside the guidelines pass the threshold on shared words; without the key concept
        # in the chunks the generator would answer from unrelated text.
        st.warning(refusal_message(coverage))
        st.stop()

    with st.spinner("Генериране на отговор..."):
        answer = generate_answer(question, chunks, model=generator)

    issues = get_checker().check(answer, chunks, question)
    attribution = get_attributor(embedding_model).attribute(answer, chunks)

    st.subheader("Отговор")
    st.markdown(highlight(answer, issues))
    if issues:
        details = ", ".join(dict.fromkeys(f"{i.text} ({i.reason})" for i in issues))
        st.warning(f"Следните подробности от отговора не се срещат в показаните откъси и може да са неточни: {details}.")
    st.caption(
        "Отговорът е генериран от езиков модел въз основа на откъсите по-долу и може да съдържа "
        "неточности. Проверявайте информацията в посочените източници."
    )
    if attribution.unsupported:
        with st.expander(f"Изречения от отговора без открито съответствие в откъсите ({len(attribution.unsupported)})"):
            for sentence in attribution.unsupported:
                st.markdown(f"- {sentence}")

    st.subheader("Източници")
    st.caption("Маркирани са изреченията от откъсите, с които са съпоставени изречения от отговора.")
    score_label = "релевантност" if hybrid else "прилика"
    for i, c in enumerate(chunks):
        spans = attribution.highlights.get(i, [])
        usage = f"маркирани изречения: {len(spans)}" if spans else "не е използван в отговора"
        with st.expander(f"{c['document']} · стр. {c['page_start']}-{c['page_end']} · {score_label} {c['score']:.3f} · {usage}"):
            st.caption(c["section"])
            st.markdown(marked(normalized_text(c), spans), unsafe_allow_html=True)
