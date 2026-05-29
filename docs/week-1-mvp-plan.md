# Week 1 MVP Plan

## Goal

By the end of Week 1, AI Research OS should support the core research chat workflow:

1. Upload a PDF.
2. Parse the paper.
3. Chunk text.
4. Generate embeddings.
5. Store vectors.
6. Retrieve relevant chunks.
7. Answer questions from the paper.

This becomes the first serious portfolio milestone: upload a paper, ask technical questions, and receive grounded answers.

## Day 1 - Project Setup

### Backend Setup

Create:

- FastAPI app
- Modular folder structure
- Environment config

Install:

- fastapi
- uvicorn
- pydantic
- python-dotenv

### Frontend Setup

Create:

- Next.js app
- Tailwind CSS setup
- shadcn/ui setup
- Chat layout

### Infrastructure Setup

Install and run:

- Docker
- Qdrant container

### Deliverables

- Running frontend
- Running backend
- Running vector database

## Day 2 - PDF Ingestion Pipeline

### PDF Upload API

Create:

- `/upload-paper`

Accept:

- PDF files

### PDF Parsing

Use:

- PyMuPDF

Extract:

- Text
- Metadata
- Page structure

### Initial Cleaning

Handle:

- Broken newlines
- References
- Duplicated spaces

### Deliverable

Upload paper and extract readable text.

## Day 3 - Chunking System

### Implement Chunking

Learn:

- Token-aware chunking
- Overlap windows
- Semantic splitting

Recommended defaults:

- 500-800 token chunks
- 100 token overlap

### Store Metadata

Each chunk stores:

- Paper ID
- Section
- Page
- Chunk index

### Deliverable

Paper converted into high-quality chunks.

## Day 4 - Embeddings + Vector Storage

### Embedding Pipeline

Use:

- BAAI BGE-small-en

Generate embeddings for chunks.

### Qdrant Integration

Store:

- Vector
- Metadata
- Text

Learn:

- Collections
- Indexing
- Similarity search

### Deliverable

Paper fully searchable semantically.

## Day 5 - Retrieval Pipeline

### Query Embedding

Convert the user question into an embedding.

### Semantic Search

Retrieve:

- Top-k relevant chunks

### Prompt Construction

Build context-aware prompts with:

- Retrieved chunks
- Instructions
- Citations

### Deliverable

Ask a question and retrieve relevant paper context.

## Day 6 - LLM Answer Generation

### Integrate Model API

Use:

- OpenRouter

Suggested models:

- Qwen
- DeepSeek
- Llama

### Generate Answers

Requirements:

- Grounded in retrieved context
- Cite sections and pages
- Avoid hallucinations

### Deliverable

Reliable paper Q&A system.

## Day 7 - Streaming Chat UI

### Build Chat Interface

Features:

- Streaming tokens
- Markdown rendering
- Citations
- PDF source references

### UX Improvements

Add:

- Loading states
- Upload progress
- Clean research workspace feel

### Deliverable

End-to-end MVP: upload paper, ask technical questions, and receive grounded answers.

## Week 1 Skills

| Skill | Why it matters |
| --- | --- |
| RAG | Core AI product pattern |
| Embeddings | Semantic understanding |
| Vector search | Memory systems |
| Prompt construction | Context engineering |
| Async APIs | Scalable backends |
| Streaming UX | Production AI interfaces |
| Retrieval evaluation | Hallucination reduction |
| AI product architecture | Startup-level systems thinking |

## MVP Success Criteria

The Week 1 MVP is complete when a user can:

- Upload a PDF paper.
- Extract readable text from it.
- Split the text into useful chunks.
- Generate embeddings for those chunks.
- Store and search vectors in Qdrant.
- Retrieve relevant context for a question.
- Generate grounded answers with citations.
- Use the workflow from a streaming chat UI.
