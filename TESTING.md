# Testing Documentation for Agentic RAG System

## Overview
This document outlines the testing strategy and coverage for the Agentic RAG system. The tests are organized into two main files:
- `tests/test_rag_system.py`: Core RAG system functionality, integration, and performance tests
- `tests/test_cli.py`: Command-line interface tests

## Test Environment Setup

### Common Test Fixtures
- **env_setup**: Sets up dummy API keys and environment variables for testing
- **clean_test_collections**: Manages and cleans up test collections
- **mock_rich_live**: Mocks Rich library displays to avoid UI conflicts
- **mock_progress**: Mocks progress bars for testing
- **mock_openai_client & patch_openai**: Mocks external OpenAI API calls and client creation
- **test_config**: Provides a test configuration for RAG system instances
- **chroma_settings**: In-memory ChromaDB settings for testing
- Additional fixtures: `pdf_dir`, `knowledge_base`, `rag_system`, etc.

## Core System Tests (`tests/test_rag_system.py`)

### Knowledge Base Tests
1. **Initialization & Configuration (`test_initialization_and_configuration`)**
   - Verifies that the KnowledgeBase initializes correctly using both OpenAI and local embedding configurations

2. **Document Processing (`test_document_processing`)**
   - Tests the ingestion and processing of PDF documents
   - Ensures that the vector store is created or updated during ingestion

3. **Collection Management (`test_collection_management`)**
   - Tests creation and persistence of multiple collections
   - Validates that collections are isolated and correctly named

### Query Processing Tests
1. **Query Workflow (`test_query_workflow`)**
   - Tests the complete query processing pipeline with document ingestion, query decomposition, and response aggregation
   - Mocks search, reasoning, and response agents to simulate the end-to-end flow

2. **Reflective Layer Evaluation (`test_reflective_layer`)**
   - Validates performance metric calculations (relevance and coherence)
   - Tests strategy adjustment based on evaluated performance

### Model / Integration Tests
1. **DeepSeek & Kluster AI Integration (`test_deepseek_integration`)**
   - Mocks the OpenAI client's chat completions to simulate responses from Kluster AI or DeepSeek
   - Verifies that the correct model and parameters are used depending on API key availability

2. **Embedding Performance (`test_embedding_performance`)**
   - Benchmarks different embedding models by comparing:
     - Precision@k
     - Recall@k
     - Retrieval time
   - Compares a local model (all-mpnet-base-v2) with OpenAI embeddings (text-embedding-3-large)

3. **Vector Store Performance (`test_vector_store_performance`)**
   - Tests the Chroma vector store for performance and optimization
   - Measures document ingestion time, query times (individual and batch), and memory usage
   - Asserts that performance remains within acceptable thresholds

## CLI Tests (`tests/test_cli.py`)

### Command Tests
1. **Help & Command Availability**
   - `test_main_menu_help`: Verifies that the main menu help is displayed
   - `test_main_menu_command`: Checks that the main menu command is accessible

2. **Core Commands**
   - `test_query_command`: Tests the direct query functionality via the CLI
   - `test_load_pdf_command`: Validates the PDF loading command help and invocation

### State Management
- Ensures that global state is reset between tests using the `reset_state` fixture
- Verifies proper cleanup of global variables and test artifacts

## Running Tests
```bash
# Run all tests with verbosity
pytest tests/ -v

# Run a specific test file
pytest tests/test_rag_system.py -v
pytest tests/test_cli.py -v

# Run a specific test
pytest tests/test_rag_system.py::test_document_processing -v
```

## Test Coverage Areas

### ✅ Well Covered
- PDF ingestion and processing workflows
- Collection management and state persistence
- Query processing pipeline with sub-query handling
- DeepSeek/Kluster AI integration and embedding model performance
- Chroma vector store performance and memory optimizations
- CLI command structure and user interaction

### 🔄 Areas for Improvement
1. **Error Handling**
   - Expand tests for network failures, invalid user inputs, and edge cases

2. **Performance Testing**
   - Include load testing with larger document sets and monitor memory usage in more detail

3. **Integration Testing**
   - Develop end-to-end workflows incorporating real API integrations on multiple platforms

4. **UI/UX Testing**
   - Enhance CLI interaction tests, including progress bar behavior and user prompt validations

## Notes for Developers
- Always run tests in isolation using the provided fixtures
- Mock external services and API calls to ensure deterministic test outcomes
- Use in-memory storage for testing vector stores
- Clean up test artifacts after each test run to avoid state contamination
- Follow the established patterns for adding and updating tests 