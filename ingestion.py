import asyncio
import os
import ssl
from typing import Any, Dict, List
import certifi
from dotenv import load_dotenv
from langchain_core import documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap, tavily_crawl, tavily_map
from openai import batches, embeddings
from logger import Colors,log_error,log_header,log_info,log_success,log_warning

load_dotenv()

# Configure SSL Context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CRT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", show_progress_bar=False, chunk_size=50, retry_min_seconds=10
)

vectorstore = PineconeVectorStore(index_name="langchain-docs",embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5,max_breadth=20,max_pages=1000)
tavily_crawl = TavilyCrawl()

async def main():
    """main async function to orchestrate everything"""
    log_header("DOCUMENTATION INGESTION PIPELINE")
    log_info(
        "Starting to crawl documentation from https://python.langchain.com",
        Colors.PURPLE
    )
        # Crawl the documentation site
    res = tavily_crawl.invoke({
        "url":"https://python.langchain.com",
        "max_depth":1,
        "extract_depth":"advanced",
        "instructions": "content on ai agents"
    })
    # Convert Tavily crawl results to LangChain Document objects
    all_docs = []
    for tavily_crawl_result_item in res["results"]:
        log_info(
            f"TavilyCrawl: Successfully crawled {tavily_crawl_result_item['url']} from documentation site"
        )
        all_docs.append(
            Document(
                page_content=tavily_crawl_result_item["raw_content"],
                metadata={"source": tavily_crawl_result_item["url"]},
            )
        )

    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f" Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000,chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks fron {len(all_docs)} documents"
    )

    # Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=500)

async def index_documents_async(documents: List[Document], batch_size: int = 50):
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f" VectorStore indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN
    )
    # Create Batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents),batch_size)
    ]

    log_info(
        f"VectorStore indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int) -> bool:
        try:
            await vectorstore.aadd_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
            return True
        except Exception as e:
            log_error(f"VectorStore indexing: Failed to add batch {batch_num} - {e}")
            return False

    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = sum(1 for r in results if r is True)
    if successful == len(batches):
        log_success(
            f"Vectorstore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_error(
            f"Vectorstore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )

if __name__ == "__main__":
    asyncio.run(main())