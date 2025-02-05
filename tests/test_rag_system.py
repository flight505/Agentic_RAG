# Standard library imports
import os
import shutil
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import psutil

# Third-party imports
import pytest

# Vector store and embedding imports
from chromadb import Settings as ChromaSettings
from rich.console import Console
from rich.table import Table

# Local imports
from agentic_rag.rag_system import (
    DeepseekQuery,
    KnowledgeBase,
    RAGConfig,
    ReflectiveLayer,
    ReflectiveRAG,
)

# Test constants
MOCK_RESPONSE = "Test response"
TEST_QUERIES = [
    "What are the main components of the system?",
    "How does error handling work?",
    "Explain the performance optimization techniques",
    "What security measures are implemented?",
    "How does the caching mechanism work?",
]

# Performance and dimension constants
OPENAI_EMBEDDING_DIMENSIONS = 3072
LOCAL_EMBEDDING_DIMENSIONS = 768
MAX_AGENT_INTERACTIONS = 3
MAX_INGESTION_TIME = 10.0  # seconds
MAX_MEMORY_IMPACT = 500  # MB

# Initialize non-interactive console for testing
console = Console(force_terminal=False)


# Test Environment Fixtures
@pytest.fixture(autouse=True)
def env_setup():
    """Set up test environment variables."""
    test_keys = {
        "DEEPSEEK_API_KEY": "dummy-deepseek-key",
        "KLUSTER_API_KEY": "dummy-kluster-key",
        "OPENAI_API_KEY": "dummy-openai-key",
    }

    # Store original values
    original_env = {key: os.environ.get(key) for key in test_keys}

    # Set test values
    for key, value in test_keys.items():
        os.environ[key] = value

    yield

    # Restore original values
    for key, value in original_env.items():
        if value:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)


@pytest.fixture(autouse=True)
def clean_test_collections():
    """Clean up test collections before and after tests."""
    test_dir = os.path.join(os.getcwd(), "knowledge_bases")
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, mode=0o777)  # Ensure full permissions
    yield
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    with patch("openai.OpenAI") as mock_openai:
        # Create mock embeddings response
        mock_embedding_object = MagicMock()
        mock_embedding_object.embedding = [0.1] * 3072  # Match the dimension of text-embedding-3-large
        
        mock_embeddings_response = MagicMock()
        mock_embeddings_response.data = [mock_embedding_object]
        
        # Create mock embeddings client
        mock_embeddings = MagicMock()
        mock_embeddings.create = MagicMock(return_value=mock_embeddings_response)
        
        # Set up the OpenAI client structure
        mock_client = MagicMock()
        mock_client.embeddings = mock_embeddings
        
        mock_client.create = mock_embeddings.create
        mock_openai.return_value = mock_client
        yield mock_client


@pytest.fixture(autouse=True)
def patch_openai(mock_openai_client):
    """Patch OpenAI client creation."""
    with patch("openai.OpenAI", return_value=mock_openai_client):
        yield


@pytest.fixture
def mock_embeddings():
    """Mock OpenAI embeddings for testing."""
    with patch("langchain_openai.embeddings.base.OpenAIEmbeddings") as mock:
        # Create a mock class that inherits from MagicMock
        class MockEmbeddings(MagicMock):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.model = "text-embedding-3-large"
                self.dimensions = OPENAI_EMBEDDING_DIMENSIONS
                self._invocation_params = {
                    "model": "text-embedding-3-large",
                    "dimensions": OPENAI_EMBEDDING_DIMENSIONS,
                }
                
                # Create a mock client
                mock_client = MagicMock()
                mock_client.embeddings = MagicMock()
                mock_client.embeddings.create = MagicMock(return_value={
                    "data": [{"embedding": [0.1] * self.dimensions}],
                    "model": self.model,
                    "usage": {"prompt_tokens": 100, "total_tokens": 100}
                })
                self.client = mock_client
            
            def embed_documents(self, texts):
                return [[0.1] * self.dimensions for _ in texts]
            
            def embed_query(self, text):
                return [0.1] * self.dimensions
            
            def _get_len_safe_embeddings(self, texts, engine=None, chunk_size=None):
                return [[0.1] * self.dimensions for _ in texts]
            
            def _embedding_with_retry(self, texts):
                if not isinstance(texts, list):
                    texts = [texts]
                return {
                    "data": [{"embedding": [0.1] * self.dimensions} for _ in texts],
                    "model": self.model,
                    "usage": {"prompt_tokens": len(texts) * 100, "total_tokens": len(texts) * 100}
                }
            
            def _tokenize(self, texts, chunk_size=None):
                if not isinstance(texts, list):
                    texts = [texts]
                return range(len(texts)), texts, None

        mock_instance = MockEmbeddings()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_progress():
    """Mock rich progress bar."""
    with patch("rich.progress.Progress.track") as mock:
        yield mock


@pytest.fixture
def test_config():
    """Create a test configuration."""
    return RAGConfig(
        llm_api_key="dummy-key",
        llm_api_base_url="https://api.kluster.ai/v1",
        embedding_provider="openai"
    )


@pytest.fixture(autouse=True)
def reset_chroma():
    """Reset ChromaDB shared system client between tests."""
    from chromadb.api.shared_system_client import SharedSystemClient

    SharedSystemClient._identifier_to_system = {}
    yield


@pytest.fixture
def chroma_settings():
    """Create in-memory Chroma settings for testing."""
    return ChromaSettings(
        is_persistent=False,
        anonymized_telemetry=False,
        allow_reset=True,  # Important for tests
        persist_directory=":memory:"  # Use in-memory storage for tests
    )


@pytest.fixture
def pdf_dir():
    """Create a test PDF directory."""
    return Path("PDFS")


@pytest.fixture
def knowledge_base(test_config, chroma_settings, mock_embeddings):
    """Create a test knowledge base with in-memory storage."""
    kb = KnowledgeBase(test_config)
    kb.chroma_settings = chroma_settings
    return kb


@pytest.fixture
def rag_system(test_config, chroma_settings, mock_embeddings):
    """Create a test RAG system with in-memory storage."""
    rag = ReflectiveRAG(test_config)
    rag.knowledge_base.chroma_settings = chroma_settings
    return rag


# Utility Functions
def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


class TestKnowledgeBase:
    """Tests for KnowledgeBase functionality and document processing."""

    def test_initialization_and_configuration(self, test_config, mock_embeddings):
        """Test knowledge base initialization with different configurations."""
        # Test OpenAI embeddings
        kb = KnowledgeBase(test_config)
        assert kb.config.embedding_provider == "openai"
        assert kb.config.openai_embedding_model == "text-embedding-3-large"
        assert kb.config.openai_dimensions == OPENAI_EMBEDDING_DIMENSIONS

        # Test fallback to local embeddings
        with patch("langchain_openai.OpenAIEmbeddings", side_effect=Exception("API Error")):
            kb = KnowledgeBase(test_config)
            assert kb.config.local_embedding_model == "sentence-transformers/all-mpnet-base-v2"
            assert kb.config.local_dimensions == LOCAL_EMBEDDING_DIMENSIONS

    def test_document_processing(self, mock_progress, knowledge_base, pdf_dir):
        """Test document ingestion and processing."""
        # Test PDF loading
        test_pdf = next(pdf_dir.glob("*.pdf"))
        knowledge_base.ingest_pdf(str(test_pdf))
        assert knowledge_base.vector_store is not None

        # Test multiple document loading
        pdf_files = list(pdf_dir.glob("*.pdf"))[:2]
        for pdf_file in pdf_files:
            knowledge_base.ingest_pdf(str(pdf_file))
        assert knowledge_base.vector_store is not None

    def test_collection_management(self, test_config, chroma_settings):
        """Test collection creation, persistence, and isolation."""
        # Test collection creation
        kb1 = KnowledgeBase(test_config, collection_name="test_collection1")
        kb2 = KnowledgeBase(test_config, collection_name="test_collection2")
        kb1.chroma_settings = chroma_settings
        kb2.chroma_settings = chroma_settings

        assert kb1.collection_name != kb2.collection_name
        assert kb1.persist_directory != kb2.persist_directory

        # Test collection persistence
        kb = KnowledgeBase(test_config, collection_name="test_persistence")
        kb.chroma_settings = chroma_settings
        assert kb.collection_name == "test_persistence"


class TestQueryProcessing:
    """Tests for query processing and RAG functionality."""

    @pytest.mark.asyncio
    async def test_query_workflow(self, mocker, rag_system, sample_pdf):
        """Test the complete query processing workflow."""
        rag_system.knowledge_base.ingest_pdf(str(sample_pdf))

        with (
            mocker.patch("agentic_rag.rag_system.ReasoningAgent") as mock_reason,
            mocker.patch("agentic_rag.rag_system.ResponseAgent") as mock_response,
            mocker.patch.object(rag_system.reflective_layer, "evaluate_response") as mock_evaluate,
        ):
            # Setup mock agents
            mock_reason_instance = mocker.MagicMock()
            mock_reason_instance.analyze.return_value = "Test analysis"
            mock_reason.return_value = mock_reason_instance

            mock_response_instance = mocker.MagicMock()
            mock_response_instance.generate.return_value = MOCK_RESPONSE
            mock_response.return_value = mock_response_instance

            mock_evaluate.return_value = 0.9

            response = await rag_system.answer_query("test query")
            assert MOCK_RESPONSE in response

    def test_reflective_layer(self, test_config):
        """Test reflective layer evaluation and strategy adjustment."""
        reflective = ReflectiveLayer(test_config)

        # Test performance evaluation
        metrics = reflective.evaluate_response("Test query", MOCK_RESPONSE, ["Test document"])
        assert isinstance(metrics, float)
        assert 0.0 <= metrics <= 1.0

        # Test strategy adjustment
        strategy = reflective.adjust_strategy(metrics)
        assert isinstance(strategy, dict)
        assert "k_retrieval" in strategy
        assert "rewrite_query" in strategy


class TestModelIntegration:
    """Tests for model integration and vector store performance."""

    @pytest.mark.skip(reason="Deepseek API is down, using Kluster AI provider; test skipped")
    def test_deepseek_integration(self, mock_openai_client, test_config):
        """Test DeepSeek and Kluster AI integration."""
        client = DeepseekQuery(test_config)
        # The direct deepseek API is not working, so this test is skipped.
        response = client.get_deepseek_response("test system", "test user")
        assert response == MOCK_RESPONSE

    def test_vector_store_performance(self, test_config, chroma_settings, pdf_dir):
        """Test Chroma vector store performance and optimizations."""
        test_config.embedding_provider = "local"

        # Initialize test metrics
        performance_metrics = {
            "memory_before": get_memory_usage(),
            "load_time": 0,
            "query_times": [],
            "batch_query_times": [],
        }

        # Initialize optimized knowledge base
        kb = KnowledgeBase(test_config, "test_performance")
        kb.chroma_settings = chroma_settings

        try:
            # Test document ingestion performance
            start_time = time.time()
            test_pdf = next(pdf_dir.glob("*.pdf"))
            kb.ingest_pdf(str(test_pdf))
            performance_metrics["load_time"] = time.time() - start_time
            performance_metrics["memory_after"] = get_memory_usage()

            # Test single query performance
            for query in TEST_QUERIES:
                start_time = time.time()
                kb.retrieve(query)
                performance_metrics["query_times"].append(time.time() - start_time)

            # Test batch query performance (recommended for production)
            start_time = time.time()
            kb.vector_store.similarity_search_by_vector(
                [kb.embeddings.embed_query(q) for q in TEST_QUERIES],
                k=test_config.k_retrieval
            )
            performance_metrics["batch_query_time"] = time.time() - start_time

            # Calculate averages
            avg_query_time = sum(performance_metrics["query_times"]) / len(TEST_QUERIES)
            batch_query_time_per_query = performance_metrics["batch_query_time"] / len(TEST_QUERIES)

            # Verify performance requirements
            assert performance_metrics["load_time"] < MAX_INGESTION_TIME, (
                f"Document ingestion time exceeds {MAX_INGESTION_TIME} seconds"
            )
            assert avg_query_time < 1.0, "Average query time exceeds 1 second"
            assert batch_query_time_per_query < avg_query_time, (
                "Batch querying should be faster than individual queries"
            )

            # Verify memory usage
            memory_impact = performance_metrics["memory_after"] - performance_metrics["memory_before"]
            assert memory_impact < MAX_MEMORY_IMPACT, f"Memory impact exceeds {MAX_MEMORY_IMPACT}MB"

            # Log performance metrics
            console.print("\n[bold]Vector Store Performance Metrics:[/bold]")
            table = Table(show_header=True, header_style="bold")
            table.add_column("Metric", style="dim")
            table.add_column("Value")

            table.add_row("Document Load Time", f"{performance_metrics['load_time']:.2f}s")
            table.add_row("Avg Single Query Time", f"{avg_query_time:.3f}s")
            table.add_row("Avg Batch Query Time", f"{batch_query_time_per_query:.3f}s")
            table.add_row("Memory Impact", f"{memory_impact:.1f}MB")

            console.print(table)

        finally:
            # Cleanup using the new reset_vector_store method
            kb.reset_vector_store()


@pytest.fixture(autouse=True)
def patch_get_len_safe_embeddings():
    from langchain_openai import OpenAIEmbeddings
    def fake_get_len_safe_embeddings(self, texts, *, engine=None, chunk_size=None):
        return [[0.1] * self.dimensions for _ in texts]
    with patch.object(OpenAIEmbeddings, "_get_len_safe_embeddings", new=fake_get_len_safe_embeddings):
        yield
