Retrieval-Augmented Generation (RAG) System
Overview
This project implements a Retrieval-Augmented Generation (RAG) system using open-source tools and local vector storage. The system enhances LLM capabilities by retrieving relevant context from a local document corpus before generating responses.

Features
Loads and processes PDF, DOCX, and TXT documents

Splits documents into manageable chunks with overlap

Generates embeddings using Sentence Transformers

Stores and retrieves embeddings using FAISS

Supports multiple retrieval strategies including similarity search and MMR

Integrates with an LLM API for response generation

Includes evaluation framework to assess RAG performance

Repository Structure
bash
Copy
Edit
.
├── rag_pipeline.py     # Main RAG pipeline integrating all components
├── rag_utils.py        # Utility functions for loading, splitting, and embedding documents
├── evaluate.py         # Evaluation framework for retrieval and generation performance
├── .env                # Contains API keys and configuration variables
├── documents/          # Folder containing input documents
├── vectorstore/        # Saved FAISS vector stores
└── README.md
Setup Instructions
1. Clone the Repository

git clone https://github.com/your-username/rag-system.git
cd rag-system
2. Create a Virtual Environment

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. Install Dependencies
pip install -r requirements.txt

4. Configure Environment Variables
Create a .env file in the root directory with the following content:
dotenv
OPENAI_API_KEY=your_openai_key_here
MODEL_NAME=gpt-3.5-turbo
EMBEDDING_MODEL_1=all-MiniLM-L6-v2
EMBEDDING_MODEL_2=multi-qa-MiniLM-L6-cos-v1
CHUNK_SIZE=500
CHUNK_OVERLAP=50
RETRIEVAL_K=5
You can add more keys as needed based on your system configuration.

5. Prepare the Document Corpus
Place your documents (PDF, DOCX, TXT) into the documents/ folder.

Usage
Build Vector Store
python rag_pipeline.py --build
Run a Query through the RAG Pipeline

python rag_pipeline.py --query "What is retrieval-augmented generation?"
Evaluate the System

python evaluate.py
Evaluation
Implemented in evaluate.py, this component uses:

Precision / Recall / F1 for retrieval performance

Heuristic metrics or qualitative inspection for answer quality

Comparisons across different retrieval settings and embedding models

Key Components
rag_utils.py
load_documents(): Loads files from documents/

split_documents(): Splits into chunks with specified overlap

generate_embeddings(): Embeds using Sentence Transformers

save_vectorstore() / load_vectorstore(): Save/load FAISS index

rag_pipeline.py
CLI tool to run the full RAG pipeline

Can build vector index, run queries, and connect to OpenAI API

Includes advanced retrieval logic (MMR, filtering)

evaluate.py
Evaluates different RAG configurations

Compares embeddings, chunk sizes, and retrieval strategies

Logs results for analysis

Challenges & Notes
FAISS requires consistent embedding dimensions between runs

LLM API limits and costs must be considered during evaluation

Text chunking must balance context completeness vs. index size

Metadata preservation is key for context-aware retrieval

References
LangChain Documentation

Sentence Transformers
FAISS GitHub
RAG Paper
