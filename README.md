# MedGuideRag

Дипломна работа: Система за извличане с генериране (RAG) за достъп до фармакотерапевтични ръководства на български език.

## Предназначение

Проектът надгражда резултатите от преддипломния проект ([Pdp](https://github.com/ccvetanska/Pdp)), в който е изграден и почистен корпус от 17 национални фармакотерапевтични ръководства. Копие на корпуса се намира в `data/processed/`, така че проектът се изпълнява самостоятелно.
MedGuideRag добавя компонентите, необходими за пълен RAG прототип:

1. Embeddings и индексиране на chunks във векторна база.
2. Retrieval компонент по зададена заявка.
3. Генериране на отговори на база извлечен контекст (prompt engineering).
4. Демонстрационно уеб приложение за въпроси и отговори.
5. Качествен анализ на резултати и оценка (TODO).

## Структура

- `src/medguiderag/ingest/` - зареждане на chunks от преддипломния корпус.
- `src/medguiderag/indexing/` - embedding модел и векторен индекс.
- `src/medguiderag/retrieval/` - компонент за извличане на релевантен контекст.
- `src/medguiderag/generation/` - генериране на отговор от извлечения контекст.
- `src/medguiderag/app/` - демонстрационно уеб приложение.
- `notebooks/` - експерименти и оценка.
- `evaluation/` - набор за разработка, скриптове за оценка и архивираните резултати зад числата в дипломната работа (виж `evaluation/README.md`).
- `reports/` - резултати от качествен анализ и retrieval benchmark.
- `docs/` - текст на дипломната работа (LaTeX).
- `data/` - данни на системата: изходният корпус (`data/processed/`) и производни данни (индекси, кешове).

## Данни

Системата ползва `data/processed/chunks.jsonl` (и `data/processed/documents.jsonl`
с текста по страници). Файловете са копие на резултата от notebook-а `Pdp/notebooks/01_extract_clean_chunk.ipynb` на преддипломния проект; виж `Pdp/docs/prediplomen_proekt.tex` за пълно описание на извличането, почистването и chunking стратегията.

## Инсталиране

Командите са за Windows и се изпълняват от основната папка на проекта (подробно описание има в Приложение 1 на дипломната работа). Необходими са Python 3.11, около 20 GB свободно място и, за работа на графичния процесор, видеокарта NVIDIA с драйвер за CUDA 12.6.

1. Генеративен модел: инсталирайте Ollama и изтеглете двата варианта на BgGPT.

   ```
   winget install --id Ollama.Ollama -e
   ollama pull hf.co/INSAIT-Institute/BgGPT-Gemma-3-4B-IT-GGUF:Q4_K_M
   ollama pull hf.co/INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0-GGUF:Q4_K_M
   ```

2. Среда на Python: библиотеките се инсталират на отделни стъпки, защото общата инсталация изчерпва паметта при разархивирането на PyTorch (поне на моята локална машина). Втората команда поставя PyTorch с CUDA и се пропуска без видеокарта NVIDIA.

   ```
   python -m venv .venv
   .venv/Scripts/python.exe -m pip install sentence-transformers
   .venv/Scripts/python.exe -m pip install "torch==2.14.0+cu126" --index-url https://download.pytorch.org/whl/cu126
   .venv/Scripts/python.exe -m pip install faiss-cpu ollama streamlit rank_bm25
   .venv/Scripts/python.exe -m pip install -e .
   ```

3. Индекси и модели: изграждат се еднократно, с връзка с интернет. След това системата работи без мрежа. Уверете се, че `data/processed/chunks.jsonl`
   съществува.

   ```
   .venv/Scripts/python.exe -m medguiderag.indexing.build_index "BAAI/bge-m3"
   .venv/Scripts/python.exe -m medguiderag.indexing.build_index "intfloat/multilingual-e5-base"
   .venv/Scripts/python.exe -m medguiderag.indexing.build_index "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
   .venv/Scripts/python.exe -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3', device='cpu')"
   ```

## Стартиране

Стартирайте `start.bat` с двоен клик. Скриптът пуска Ollama, ако не работи, ограничава броя на нишките на числовите библиотеки и стартира приложението на адрес http://localhost:8501, достъпно само от локалната машина. Първият въпрос отнема около 35 секунди, докато се заредят моделите. Приложението се спира с Ctrl+C в отворения прозорец.

Ръчно стартиране:

```
.venv/Scripts/python.exe -m streamlit run src/medguiderag/app/ui.py
```

Възпроизвеждане на оценката: виж `evaluation/README.md`.
