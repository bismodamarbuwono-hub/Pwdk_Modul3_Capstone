import argparse
import os
import re

import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CSV_PATH = "Resume.csv"
COLLECTION = "resumes"
EMBED_MODEL = "text-embedding-3-small"  # 1536 dimensions
VECTOR_SIZE = 1536
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MIN_CHARS = 200  # skip near-empty resumes
BATCH = 100


def clean(text: str) -> str:
    """Collapse the many spaces/newlines in the scraped resume text."""
    return re.sub(r"\s+", " ", str(text)).strip()


def load_documents(limit=None):
    df = pd.read_csv(CSV_PATH, usecols=["ID", "Resume_str", "Category"])
    df = df.dropna(subset=["Resume_str"])
    df["Resume_str"] = df["Resume_str"].map(clean)
    df = df[df["Resume_str"].str.len() >= MIN_CHARS]
    if limit:
        per_category = max(1, limit // df["Category"].nunique())
        df = df.groupby("Category").head(per_category).head(limit)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    docs = []
    for row in df.itertuples(index=False):
        pieces = splitter.split_text(row.Resume_str)
        for i, piece in enumerate(pieces):
            docs.append(
                Document(
                    page_content=piece,
                    metadata={
                        "resume_id": str(row.ID),
                        "category": row.Category,
                        "chunk_index": i,
                    },
                )
            )
    return df, docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--recreate", action="store_true")
    args = ap.parse_args()

    df, docs = load_documents(args.limit)
    print(f"Resumes kept: {len(df)}")
    print(f"Chunks created: {len(docs)}  (avg {len(docs) / len(df):.1f} per resume)")
    est_tokens = sum(len(d.page_content) for d in docs) / 4
    print(f"Approx. embedding tokens: {est_tokens:,.0f}  (~${est_tokens / 1e6 * 0.02:.3f})")
    print("Sample chunk:", docs[0].metadata, "->", docs[0].page_content[:150], "...")
    if args.dry_run:
        return

    load_dotenv()
    from langchain_openai import OpenAIEmbeddings
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient, models

    client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=os.environ["OPENAI_API_KEY"])

    exists = client.collection_exists(COLLECTION)
    if exists and args.recreate:
        client.delete_collection(COLLECTION)
        exists = False
    if not exists:
        client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )
        for field in ("metadata.category", "metadata.resume_id"):
            client.create_payload_index(
                COLLECTION, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD
            )

    store = QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=embeddings)
    for start in range(0, len(docs), BATCH):
        store.add_documents(docs[start : start + BATCH])
        print(f"Uploaded {min(start + BATCH, len(docs))}/{len(docs)}")

    print("Done. Points in collection:", client.count(COLLECTION).count)


if __name__ == "__main__":
    main()