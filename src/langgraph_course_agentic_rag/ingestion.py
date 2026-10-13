import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), os.getenv("AZURE_TOKEN_PROVIDER")
)
embeddings = OpenAIEmbeddings(
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    model=EMBEDDING_MODEL,
    api_key=token_provider,
)

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [UnstructuredLoader(web_url=url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     embedding=embeddings,
#     collection_name="langchain_course_agentic_rag",
#     persist_directory="./.chroma_db",
# )

retriever = Chroma(
    collection_name="langchain_course_agentic_rag",
    embedding_function=embeddings,
    persist_directory="./.chroma_db",
).as_retriever()
