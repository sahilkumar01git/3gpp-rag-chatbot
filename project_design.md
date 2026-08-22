# 3GPP RAG Chatbot — Project Design

## 1. Project Overview

The 3GPP RAG Chatbot is a retrieval-augmented generation system designed to answer technical questions using 3GPP specification documents as the source of truth.

The primary objective is to generate grounded, citation-backed answers while minimizing unsupported or hallucinated information.

## 2. Problem Statement

3GPP specifications contain large amounts of highly technical and structured information distributed across multiple documents and clauses.

A general-purpose LLM may generate plausible but unsupported answers when asked about such information.

This project addresses this problem by retrieving relevant evidence from 3GPP specifications before generating an answer and applying verification and confidence checks before returning the response.

## 3. System Architecture

The system follows this pipeline:

3GPP PDFs
→ PDF Parsing
→ Document Processing
→ Chunking
→ Embedding Generation
→ FAISS Vector Search
→ Cross-Encoder Reranking
→ Context Selection
→ LLM Generation
→ Claim Verification
→ Confidence Evaluation
→ Citation-Backed Answer / Refusal

## 4. Document Ingestion

PDF documents are processed using PDF extraction utilities.

The extracted content is divided into meaningful chunks while retaining metadata such as:

- Specification number
- Version
- Clause information
- Page information
- Source document

This metadata is later used for traceability and citations.

## 5. Embeddings and Vector Search

Text chunks are converted into vector embeddings using a sentence-transformer embedding model.

FAISS is used as the vector database for efficient similarity-based retrieval.

For a user query, the system retrieves the most relevant chunks from the indexed specification corpus.

## 6. Reranking

Initial vector retrieval can return semantically similar but less relevant chunks.

A cross-encoder reranker is therefore used to score the retrieved passages against the user query and improve the quality of the final evidence set.

## 7. Generation

The selected evidence is passed to the language model along with instructions to answer using the supplied context.

The system is designed to avoid relying on unsupported model knowledge when answering technical questions.

## 8. Hallucination Control

The system includes multiple mechanisms to reduce unsupported responses:

- Evidence-based retrieval
- Cross-encoder reranking
- Claim-level verification
- Confidence scoring
- Refusal/degraded response behavior
- Source citations

Claims that cannot be sufficiently supported by retrieved evidence can be rejected instead of being presented as factual information.

## 9. Citations

Generated responses include source information where available, including specification, version, clause, and page metadata.

This allows users to trace an answer back to the underlying technical document.

## 10. Conversation Memory

The application maintains conversation context so that follow-up questions can be interpreted using previous interaction history.

Memory is used to improve conversational usability while the retrieved specification evidence remains the primary factual source.

## 11. Backend

FastAPI provides the backend API layer.

The backend handles:

- Query processing
- Retrieval
- Reranking
- Generation
- Verification
- Response construction

## 12. Frontend

A web-based interface is provided for interacting with the RAG chatbot.

Users can submit technical questions and receive answers together with supporting source information.

## 13. Evaluation

The project includes an evaluation dataset containing both normal in-scope questions and adversarial/out-of-scope questions.

The evaluation measures retrieval and response quality using metrics such as:

- Recall@K
- MRR
- Keyword coverage
- Refusal behavior
- Hallucination-related checks
- Response latency

## 14. Error Handling

The application includes handling for common runtime conditions such as:

- API failures
- Missing configuration
- Retrieval failures
- Model/API errors
- Unsupported questions
- Insufficient evidence

The system is designed to fail safely rather than confidently provide unsupported technical information.

## 15. Security and Configuration

API credentials are supplied through environment variables.

Secrets are excluded from source control using `.gitignore`.

An `.env.example` file is provided to document the required configuration without exposing credentials.

## 16. Limitations

The quality of responses depends on the documents available in the indexed corpus.

If the required information is not present in the available specifications, the system may refuse to answer rather than rely on unsupported external knowledge.

## 17. Future Improvements

Potential improvements include:

- Larger 3GPP document coverage
- Improved clause-aware chunking
- Additional retrieval strategies
- More extensive evaluation datasets
- Advanced factual consistency checks
- Production-scale observability and monitoring

## 18. Technology Stack

- Python
- FastAPI
- FAISS
- Sentence Transformers
- Cross-Encoder
- Groq / LLM
- PDFPlumber
- Streamlit/Web UI
- Pytest