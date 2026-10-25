# Resume Screening Assistant

A multi-agent RAG chatbot that helps recruiters find, summarize, and compare candidates from a resume database. Built for Capstone Project Module 3.

Live app: https://pwdkmodul3capstone-nxt9fxhopdnxyvbp9rpmbk.streamlit.app/

## What it does

Ask in plain language, for example:

- "Find candidates with Python and SQL experience"
- "Summarize the first one"
- "Compare it with the second one"
- "Find an experienced chef with banquet and menu planning experience"

The bot answers only from the resume database. It cites the resume ID and category of every candidate, says clearly when no resume matches, and politely declines questions that are not about the resumes.

## How it works

```
User (Streamlit)
      |
Supervisor agent
      |-- Candidate Search agent --> search_resumes (RAG tool) --\
      |                                                           +--> Qdrant Cloud
      |-- Resume Analyst agent  --> get_resume_by_id -----------/
```

1. `ingest.py` cleans the resumes, splits them into chunks (1,000 characters, 150 overlap), creates embeddings with OpenAI `text-embedding-3-small`, and stores them in Qdrant with the metadata `resume_id`, `category`, and `chunk_index`.
2. The **Supervisor agent** reads the question and the last 6 chat messages, then calls the right specialist.
3. The **Candidate Search agent** uses the RAG tool `search_resumes` (semantic search, optional category filter).
4. The **Resume Analyst agent** uses `get_resume_by_id` to read full resumes and summarize or compare them.
5. The Streamlit app shows the answer, the retrieved resume text (sources), the tool calls, token usage, and the estimated cost in rupiah.
6. Every run is traced in **Langfuse**, and all messages of one chat share one session.

## Tech stack

- Python, Streamlit (deployed on Streamlit Community Cloud)
- LangChain and LangGraph (agents and tools)
- OpenAI: `gpt-4o-mini` (chat) and `text-embedding-3-small` (embeddings)
- Qdrant Cloud (vector database)
- Langfuse (tracing)

## Dataset

[Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset): 2,400+ resumes in 24 job categories. After removing near-empty resumes, 2,483 resumes (18,086 chunks) were indexed. The data file is not included in this repository.

## Files

| File | Purpose |
|------|---------|
| `ingest.py` | Loads the CSV, chunks, embeds, and uploads to Qdrant (run once locally) |
| `agents.py` | RAG tools, specialist agents, supervisor, prompts, token tracking |
| `app.py` | Streamlit chat interface |
| `requirements.txt` | Python packages |

## Run it yourself

1. Install packages: `python -m pip install -r requirements.txt` (plus `pandas` and `langchain-text-splitters` for `ingest.py`).
2. Create a `.env` file with your own keys:
   ```
   OPENAI_API_KEY=...
   QDRANT_URL=...
   QDRANT_API_KEY=...
   LANGFUSE_PUBLIC_KEY=...
   LANGFUSE_SECRET_KEY=...
   LANGFUSE_BASE_URL=...
   ```
3. Put `Resume.csv` in the same folder and run `python ingest.py --recreate` once to fill Qdrant.
4. Start the app: `python -m streamlit run app.py`.

On Streamlit Community Cloud, put the same six values in the app's **Secrets** (TOML format, values in quotes).

## Notes

- API keys are never stored in this repository (`.env` is git-ignored).
- The resumes come from a public dataset and may contain personal details, so treat the output with care.