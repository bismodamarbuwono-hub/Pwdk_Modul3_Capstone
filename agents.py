"""
agents.py - RAG tools + multi-agent supervisor for the Resume Screening Assistant.

Architecture (supervisor with agents as tools):

    Supervisor agent
      |- candidate_search_agent  -> uses search_resumes (RAG over Qdrant)
      |- resume_analyst_agent    -> uses get_resume_by_id (Qdrant lookup)

Works locally (reads .env) and on Streamlit Cloud (reads st.secrets).
Test it from the terminal:  python agents.py
"""
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler, UsageMetadataCallbackHandler
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.prebuilt import create_react_agent
from qdrant_client import models

load_dotenv()

COLLECTION = "resumes"
LLM_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
MAX_HISTORY = 6  # how many past messages (3 user + 3 assistant) the agent sees

# gpt-4o-mini prices in USD per 1M tokens, and an approximate exchange rate
PRICE_IN_USD, PRICE_OUT_USD, USD_TO_IDR = 0.15, 0.60, 17_000

CATEGORIES = [
    "ACCOUNTANT", "ADVOCATE", "AGRICULTURE", "APPAREL", "ARTS", "AUTOMOBILE", "AVIATION",
    "BANKING", "BPO", "BUSINESS-DEVELOPMENT", "CHEF", "CONSTRUCTION", "CONSULTANT",
    "DESIGNER", "DIGITAL-MEDIA", "ENGINEERING", "FINANCE", "FITNESS", "HEALTHCARE", "HR",
    "INFORMATION-TECHNOLOGY", "PUBLIC-RELATIONS", "SALES", "TEACHER",
]

# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------
SUPERVISOR_PROMPT = f"""You are the supervisor of a resume screening assistant that helps recruiters.
You can ONLY help with questions about the resume database: finding candidates, summarizing a
resume, and comparing candidates. You have two specialist tools:

1. candidate_search_agent - finds resumes that match skills, roles, or experience.
2. resume_analyst_agent - reads specific resumes BY resume_id to summarize or compare them.

Rules:
- Always call a tool before answering about candidates. Never use your own knowledge and never
  invent resumes, names, or experience.
- To compare or summarize candidates you already mentioned, use the conversation history to find
  their resume_id values and pass those IDs to resume_analyst_agent. If the user says "the first
  one" or "those two", resolve it from the history.
- Only present a candidate if the tool results show they meet the user's key requirements. If the tools
  report no match, tell the user clearly that no matching resume was found. Never turn near-misses into
  matches, and never add details that are not in the tool results.
- Always mention the resume_id (and category) of every candidate you refer to.
- If the question is not about the resumes (general knowledge, coding help, chit-chat), politely
  decline and say what you can do instead.
- Job categories in the database: {", ".join(CATEGORIES)}.
- Answer in the same language the user writes in. Keep answers concise and well organized."""

SEARCH_PROMPT = f"""You are a resume search specialist. Use the search_resumes tool to find resumes
that match the request. If the request clearly belongs to one job category, pass it as the
category argument (valid values: {", ".join(CATEGORIES)}); otherwise leave it empty.

IMPORTANT: the search tool ALWAYS returns the closest chunks, even when none of them truly match.
A resume matches only if its text shows ALL the key requirements of the request (for example the job
title AND the skill or experience asked for). A resume that merely shares a word with the request does
NOT match: a company or place that happens to be named "Mars" is not experience on Mars, and a chef is
not an astronaut. Before listing a resume, check its text against every requirement and list only the
resumes that pass. For each one, give the resume_id, category, and a one-line reason it matches, based
ONLY on the tool results.
If no resume passes, reply that no resume in the database matches, and briefly say what you searched
for. Do NOT list near-misses as matches. Mention partial matches only if the user explicitly asks for
the closest or similar candidates, and then say clearly what is missing."""

ANALYST_PROMPT = """You are a resume analyst. Use the get_resume_by_id tool to read each resume the
request mentions, then summarize it or compare candidates as requested. Base everything ONLY on the
resume text returned by the tool. If a resume id is not found, say so. Always include the resume_id."""


# ----------------------------------------------------------------------------
# Secrets, model, and vector store
# ----------------------------------------------------------------------------
def get_secret(name: str) -> str:
    """Read from Streamlit secrets when deployed, otherwise from .env / environment."""
    try:
        import streamlit as st

        return st.secrets[name]
    except Exception:
        return os.environ[name]


_cache = {}


def get_llm():
    if "llm" not in _cache:
        _cache["llm"] = ChatOpenAI(model=LLM_MODEL, api_key=get_secret("OPENAI_API_KEY"), temperature=0)
    return _cache["llm"]


def get_store():
    if "store" not in _cache:
        embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=get_secret("OPENAI_API_KEY"))
        _cache["store"] = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=COLLECTION,
            url=get_secret("QDRANT_URL"),
            api_key=get_secret("QDRANT_API_KEY"),
        )
    return _cache["store"]


# ----------------------------------------------------------------------------
# Tracking which tools ran and what the RAG tools returned (for the UI)
# ----------------------------------------------------------------------------
RAG_TOOLS = ("search_resumes", "get_resume_by_id")


class ToolTrace(BaseCallbackHandler):
    """Records every tool call, plus the raw documents returned by the RAG tools."""

    def __init__(self):
        self.names = {}
        self.calls = []    # every tool call, including the sub-agents
        self.sources = []  # raw output of the RAG tools

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        name = (serialized or {}).get("name") or kwargs.get("name") or "unknown"
        self.names[run_id] = name
        self.calls.append({"tool": name, "input": str(input_str)})

    def on_tool_end(self, output, *, run_id, **kwargs):
        name = self.names.get(run_id)
        if name in RAG_TOOLS:
            self.sources.append({"tool": name, "output": str(getattr(output, "content", output))})


# ----------------------------------------------------------------------------
# Build the multi-agent system
# ----------------------------------------------------------------------------
def _category_filter(category: Optional[str]):
    if not category:
        return None
    cat = category.strip().upper()
    if cat not in CATEGORIES:
        return None
    return models.Filter(
        must=[models.FieldCondition(key="metadata.category", match=models.MatchValue(value=cat))]
    )


def _format_doc(doc) -> str:
    m = doc.metadata
    return f"[resume_id={m.get('resume_id')} | category={m.get('category')}]\n{doc.page_content}"


def build_supervisor(k: int = 5):
    llm = get_llm()
    store = get_store()

    # ---- RAG tools -------------------------------------------------------
    @tool
    def search_resumes(query: str, category: Optional[str] = None) -> str:
        """Semantic search over the resume database. Use it to find resumes matching skills,
        job titles, or experience. Optionally pass a job category to narrow the search."""
        docs = store.similarity_search(query, k=k, filter=_category_filter(category))
        if not docs:
            return "No matching resumes found."
        return "\n\n---\n\n".join(_format_doc(d) for d in docs)

    @tool
    def get_resume_by_id(resume_id: str) -> str:
        """Fetch the full text of ONE resume using its resume_id."""
        flt = models.Filter(
            must=[models.FieldCondition(
                key="metadata.resume_id", match=models.MatchValue(value=str(resume_id).strip()))]
        )
        points, _ = store.client.scroll(
            collection_name=COLLECTION, scroll_filter=flt, limit=50,
            with_payload=True, with_vectors=False,
        )
        if not points:
            return f"No resume found with id {resume_id}."
        points.sort(key=lambda p: p.payload["metadata"]["chunk_index"])
        category = points[0].payload["metadata"]["category"]
        text = " ".join(p.payload["page_content"] for p in points)
        return f"[resume_id={resume_id} | category={category}]\n{text}"

    # ---- Specialist agents -----------------------------------------------
    search_agent = create_react_agent(llm, [search_resumes], prompt=SEARCH_PROMPT)
    analyst_agent = create_react_agent(llm, [get_resume_by_id], prompt=ANALYST_PROMPT)

    # ---- Wrap the agents as tools for the supervisor ---------------------
    # Passing `config` on lets token counting and tracing see the nested calls too.
    @tool
    def candidate_search_agent(request: str, config: RunnableConfig) -> str:
        """Find candidates whose resumes match skills, job titles, or experience.
        Input: a clear description of the kind of candidate to look for."""
        result = search_agent.invoke({"messages": [{"role": "user", "content": request}]}, config=config)
        return result["messages"][-1].content

    @tool
    def resume_analyst_agent(request: str, config: RunnableConfig) -> str:
        """Read specific resumes by resume_id to summarize one or compare several.
        Input: the task plus the resume_id values, e.g. 'Compare resumes 12345 and 67890'."""
        result = analyst_agent.invoke({"messages": [{"role": "user", "content": request}]}, config=config)
        return result["messages"][-1].content

    return create_react_agent(
        llm, [candidate_search_agent, resume_analyst_agent], prompt=SUPERVISOR_PROMPT
    )


# ----------------------------------------------------------------------------
# Main entry point used by the Streamlit app
# ----------------------------------------------------------------------------
def ask(question: str, history=None, k: int = 5, extra_callbacks=None, session_id=None) -> dict:
    """
    history: list of {"role": "user" | "assistant", "content": str}, NOT including `question`.
    session_id: one id per chat; Langfuse groups all turns with the same id into one session.
    Returns the answer plus sources, tool calls, and token/cost info.
    """
    supervisor = build_supervisor(k)
    messages = [
        {"role": m["role"], "content": m["content"]} for m in (history or [])[-MAX_HISTORY:]
    ]
    messages.append({"role": "user", "content": question})

    usage = UsageMetadataCallbackHandler()
    trace = ToolTrace()
    callbacks = [usage, trace] + list(extra_callbacks or [])
    config = {"callbacks": callbacks}
    if session_id:
        config["metadata"] = {"langfuse_session_id": session_id}
    result = supervisor.invoke({"messages": messages}, config=config)

    tokens_in = sum(u.get("input_tokens", 0) for u in usage.usage_metadata.values())
    tokens_out = sum(u.get("output_tokens", 0) for u in usage.usage_metadata.values())
    price_idr = USD_TO_IDR * (tokens_in * PRICE_IN_USD + tokens_out * PRICE_OUT_USD) / 1_000_000

    return {
        "answer": result["messages"][-1].content,
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "price_idr": price_idr,
        "tool_calls": trace.calls,
        "sources": trace.sources,
    }


if __name__ == "__main__":
    # Simple terminal chat to test follow-up questions before building the Streamlit UI.
    chat = []
    print("Resume assistant (type 'exit' to quit)")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        r = ask(q, chat)
        print("\nAI:", r["answer"])
        print(f"\n[tools: {[c['tool'] for c in r['tool_calls']]} | "
              f"tokens in={r['input_tokens']} out={r['output_tokens']} | Rp {r['price_idr']:.1f}]")
        chat += [{"role": "user", "content": q}, {"role": "assistant", "content": r["answer"]}]
