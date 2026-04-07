# LlamaIndex Kawn Integration

This repository provides LlamaIndex wrappers for the **Kawn AI SDK**, allowing you to seamlessly integrate state-of-the-art Arabic AI models into your LlamaIndex pipelines.

This integration includes:
* **`KawnEmbedding`**: High-quality document and query embedding via Kawn's Tbyaan models, optimized for Arabic and Islamic content.
* **`BaseerReader`**: A sophisticated OCR data reader powered by Kawn's Baseer API, extracting structural markdown and exceptional Arabic text from documents (PDFs, Images).

## Installation

```bash
pip install llama-index-kawn
```

## Setup

The easiest way to configure the integration is by setting your Kawn API key as an environment variable. Alternatively, you can pass it directly to the instances.

```bash
export KAWN_API_KEY="your_api_key_here" # Or MISRAJ_API_KEY="your_api_key_here"
```

## Usage

### 1. KawnEmbedding

`KawnEmbedding` allows you to generate robust vector representations of queries and documents. You can use it as the default embedding model in your `LlamaIndex` settings or interface with it directly to generate embeddings.

```python
from llama_index_integration.embeddings.kawn import KawnEmbedding
from llama_index.core import Settings

# 1. Initialize the Kawn Embedding model
# It automatically picks up the KAWN_API_KEY environment variable.
embed_model = KawnEmbedding(
    model_name="tbyaan/islamic-embedding-tbyaan-v1",
    dimensions=768 # Optional: specify desired output dimension
)

# 2. Set as the default embeddings model in LlamaIndex globally
Settings.embed_model = embed_model

# 3. Direct Embedding Generation
query_embedding = embed_model.get_query_embedding("ما هو تفسير سورة الفاتحة؟")
print(f"Query embedding dimension: len({len(query_embedding)})")

text_batch = ["النص الأول", "النص الثاني"]
batch_embeddings = embed_model.get_text_embedding_batch(text_batch)
```

### 2. BaseerReader

`BaseerReader` leverages Baseer's highly accurate OCR service to read documents (such as `.pdf`, `.png`, `.jpg`) and instantly convert them into LlamaIndex `Document` objects. It is optimized for structural data extraction and advanced Arabic layouts.

```python
from llama_index_integration.readers.kawn import BaseerReader

# 1. Initialize the Baseer Reader
reader = BaseerReader(
    model="baseer/baseer-v2", # Default OCR model
    # Optional dictionary of OCR configuration parameters
  )

# 2. Load and process a local file
file_path = "sample_book.pdf"

# By default, it returns a list of Documents (one per page). 
# Set one_text_result=True to merge everything into a single LlamaIndex Document.
documents = reader.load_data(
    file_path=file_path, 
    one_text_result=False,
    extra_info={"category": "Islamic History"} # Appended to metadata
)

for doc in documents:
    print(f"Page {doc.metadata['page_index']}:")
    print(doc.text[:200]) # Print the first 200 characters of the page
```

### 3. End-to-End RAG Pipeline (LlamaIndex Vector Store)

Here is how you can combine both `BaseerReader` and `KawnEmbedding` to read an Arabic PDF, embed it, store it in a vector database, and query it.

```python
from llama_index.core import VectorStoreIndex, Settings
from llama_index_integration.embeddings.kawn import KawnEmbedding
from llama_index_integration.readers.kawn import BaseerReader

# 1. Set Kawn as the global embedding model
Settings.embed_model = KawnEmbedding()

# 2. Extract text and structure from a complex Arabic document
reader = BaseerReader()
documents = reader.load_data("complex_arabic_document.pdf")

# 3. Build a Vector Store Index (In-memory by default, easily swapped for Chroma/Qdrant)
index = VectorStoreIndex.from_documents(documents)

# 4. Query the document
query_engine = index.as_query_engine()
response = query_engine.query("ما هي الاستنتاجات الرئيسية في هذا التقرير؟")
print(response)
```

### 4. Using BaseerReader with LangChain

While `BaseerReader` is built as a native LlamaIndex integration, you can easily use its exceptional OCR capabilities to extract text and feed it directly into a LangChain conversational pipeline.

```python
from llama_index_integration.readers.kawn import BaseerReader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. Extract pure text using BaseerReader
reader = BaseerReader()
# one_text_result=True is useful here to pass a single context block to the LLM
documents = reader.load_data("sample_book.pdf", one_text_result=True)
document_text = documents[0].text

# 2. Setup LangChain LLM and Prompt
llm = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages([
    ("system", "أنت مساعد ذكي. أجب على أسئلة المستخدم بناءً على السياق التالي فقط:\n\n{context}"),
    ("human", "{question}")
])

# 3. Create the chain and ask a question based on the document
chain = prompt | llm
response = chain.invoke({
    "context": document_text,
    "question": "لخص أهم النقاط المذكورة في هذا النص."
})

print(response.content)
```

### Async Support

Both `KawnEmbedding` and `BaseerReader` natively support non-blocking asynchronous operations, making them ideal for high-throughput batching or async web servers (like FastAPI).

```python
import asyncio
from llama_index_integration.embeddings.kawn import KawnEmbedding
from llama_index_integration.readers.kawn import BaseerReader

async def main():
    embed_model = KawnEmbedding()
    reader = BaseerReader()

    # Asynchronous Embedding Generation
    query_embed = await embed_model.aget_query_embedding("مرحبا بك في منصة كون")
    
    # Asynchronous OCR Request (handles background polling for you)
    docs = await reader.aload_data("sample_document.pdf")

asyncio.run(main())
```