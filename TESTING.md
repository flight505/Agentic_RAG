# Testing Documentation for Agentic RAG System

## Overview
This document outlines the testing strategy and coverage for the Agentic RAG system. The tests are organized into two main files:
- `tests/test_rag_system.py`: Core RAG system functionality tests
- `tests/test_cli.py`: Command-line interface tests

## Test Environment Setup

### Common Test Fixtures
- `env_setup`: Sets up dummy API keys for testing
- `clean_test_collections`: Manages test collections cleanup
- `mock_rich_live`: Prevents conflicts with Rich library displays
- `mock_progress`: Mocks progress bars for testing
- `mock_api` and `mock_client`: Mock OpenAI API calls
- `test_config`: Provides test configuration
- `chroma_settings`: In-memory ChromaDB settings for testing

## Core System Tests (`test_rag_system.py`)

### Knowledge Base Tests
1. **PDF Loading (`test_pdf_loading`)**
   - Tests the ingestion of PDF documents
   - Verifies vector store initialization

2. **Collection Management**
   - `test_collection_creation`: Tests creating named collections
   - `test_collection_persistence`: Verifies data persistence
   - `test_collection_isolation`: Ensures collections remain isolated
   - `test_multiple_collections`: Tests managing multiple collections
   - `test_knowledge_base_reuse`: Verifies handling multiple PDF loads

### Query Processing Tests
1. **Query Processing Pipeline (`test_query_processing`)**
   - Tests the complete query processing workflow
   - Mocks reasoning and response agents
   - Verifies sub-query handling
   - Tests response generation and combination

2. **Performance Metrics (`test_performance_metrics`)**
   - Tests the reflective layer's evaluation capabilities
   - Verifies performance metric calculations
   - Tests strategy adjustment based on performance

### Model Tests
1. **Embedding Model Performance (`test_embedding_performance`)**
   - Compares different embedding models:
     - Old: sentence-transformers/all-mpnet-base-v2 (768 dimensions)
     - New: Salesforce/SFR-Embedding-Mistral (4096 dimensions)
   - Measures and compares:
     - Precision@k
     - Recall@k
     - Retrieval time

2. **DeepSeek Query Tests (`test_deepseek_query`)**
   - Tests Kluster AI integration
   - Verifies fallback mechanism to DeepSeek
   - Tests response handling

### Reflective Layer Tests
1. **Core Functionality (`test_reflective_layer`)**
   - Tests performance evaluation
   - Tests strategy adjustment
   - Verifies metric ranges and types

2. **RAG with Collections (`test_rag_with_collections`)**
   - Tests RAG system with multiple collections
   - Verifies query processing across collections
   - Tests agent coordination

## CLI Tests (`test_cli.py`)

### Command Tests
1. **Help Commands**
   - `test_main_menu_help`: Tests main menu help display
   - `test_main_menu_command`: Tests main menu command availability

2. **Core Commands**
   - `test_query_command`: Tests direct query functionality
   - `test_load_pdf_command`: Tests PDF loading command

### State Management
- Tests maintain clean state between runs using the `reset_state` fixture
- Verifies proper cleanup of global variables

## Running Tests
```bash
# Run all tests with verbosity
pytest tests/ -v

# Run specific test file
pytest tests/test_rag_system.py -v
pytest tests/test_cli.py -v

# Run specific test
pytest tests/test_rag_system.py::test_pdf_loading -v
```

## Test Coverage Areas

### ✅ Well Covered
- PDF ingestion and processing
- Collection management
- Query processing pipeline
- Model performance comparison
- CLI command structure

### 🔄 Areas for Improvement
1. **Error Handling**
   - Add more edge cases
   - Test network failures
   - Test invalid input handling

2. **Performance Testing**
   - Add load testing
   - Test with larger document sets
   - Memory usage monitoring

3. **Integration Testing**
   - End-to-end workflows
   - Real API integration tests
   - Cross-platform testing

4. **UI/UX Testing**
   - CLI interaction testing
   - Progress bar functionality
   - User input validation

## Notes for Developers
- Always run tests in isolation using the provided fixtures
- Mock external services and API calls
- Use in-memory databases for testing
- Clean up test artifacts after each test
- Follow the existing pattern for adding new tests 