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
from rich.layout import Layout
from rich.panel import Panel
from rich.style import Style
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
MAX_INGESTION_TIME = 15.0  # seconds - increased to account for Marker PDF processing
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
                mock_client.embeddings.create = MagicMock(
                    return_value={
                        "data": [{"embedding": [0.1] * self.dimensions}],
                        "model": self.model,
                        "usage": {"prompt_tokens": 100, "total_tokens": 100},
                    }
                )
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
                    "usage": {"prompt_tokens": len(texts) * 100, "total_tokens": len(texts) * 100},
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
    return RAGConfig(llm_api_key="dummy-key", llm_api_base_url="https://api.kluster.ai/v1", embedding_provider="openai")


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
        persist_directory=":memory:",  # Use in-memory storage for tests
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

    def test_query_workflow(self, mocker, rag_system, pdf_dir):
        """Test the complete query processing workflow."""
        test_pdf = next(pdf_dir.glob("*.pdf"))
        rag_system.knowledge_base.ingest_pdf(str(test_pdf))

        # Setup mock agents
        # Create mock agents
        mock_search = mocker.MagicMock()
        mock_reason = mocker.MagicMock()
        mock_reason.analyze.return_value = "Test analysis"
        mock_response = mocker.MagicMock()
        mock_response.generate.return_value = MOCK_RESPONSE

        # Patch the agents dictionary
        mocker.patch.object(
            rag_system.orchestrator, "agents", {"search": mock_search, "reason": mock_reason, "response": mock_response}
        )

        # Patch evaluate_response
        mocker.patch.object(rag_system.reflective_layer, "evaluate_response", return_value=0.9)

        # Run the test
        response = rag_system.answer_query("test query")
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

    # Deepseek API is down, using Kluster AI provider
    def test_deepseek_integration(self, mock_openai_client, test_config):
        """Test DeepSeek and Kluster AI integration."""
        # Setup mock response
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content=MOCK_RESPONSE))]
        mock_openai_client.chat.completions.create.return_value = mock_completion

        # Test with Kluster AI
        test_config.llm_api_base_url = "https://api.kluster.ai/v1"
        client = DeepseekQuery(test_config, client=mock_openai_client)
        client.kluster_key = "dummy-kluster-key"
        client.use_kluster = True

        response = client.get_deepseek_response("test system", "test user")
        assert response == MOCK_RESPONSE
        mock_openai_client.chat.completions.create.assert_called_with(
            model="deepseek-ai/DeepSeek-R1",
            messages=[
                {"role": "system", "content": "test system"},
                {"role": "user", "content": "test user"},
            ],
            max_tokens=1024,
            temperature=0.7,
            stream=False,
        )

        # Test DeepSeek path
        mock_openai_client.chat.completions.create.reset_mock()
        client.use_kluster = False
        response = client.get_deepseek_response("test system", "test user")
        assert response == MOCK_RESPONSE
        mock_openai_client.chat.completions.create.assert_called_with(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "test system"},
                {"role": "user", "content": "test user"},
            ],
            max_tokens=1024,
            temperature=0.7,
            stream=False,
        )

    def test_embedding_performance(self, test_config, chroma_settings, pdf_dir):
        """Compare performance of different embedding models."""
        # Create layout for rich output
        layout = Layout()
        layout.split_column(
            Layout(Panel.fit("Embedding Models Performance Comparison", style="bold magenta"), size=3),
            Layout(name="main"),
        )

        # Test configuration for local model
        test_config.embedding_provider = "local"
        test_config.local_embedding_model = "sentence-transformers/all-mpnet-base-v2"
        test_config.local_dimensions = 768

        # Initialize knowledge base with local model
        kb_local = KnowledgeBase(test_config, "test_performance_local")
        kb_local.chroma_settings = chroma_settings

        # Test configuration for OpenAI model
        test_config.embedding_provider = "openai"
        test_config.openai_embedding_model = "text-embedding-3-large"
        test_config.openai_dimensions = 3072
        kb_openai = KnowledgeBase(test_config, "test_performance_openai")
        kb_openai.chroma_settings = chroma_settings

        # Performance metrics tables
        local_metrics_table = Table(title="Local Model (all-mpnet-base-v2)", show_header=True)
        openai_metrics_table = Table(title="OpenAI Model (text-embedding-3-large)", show_header=True)

        for table in [local_metrics_table, openai_metrics_table]:
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right")

        try:
            # Benchmark both configurations
            test_pdf = next(pdf_dir.glob("*.pdf"))

            # Test local model
            console.print(Panel("Testing local model...", style="blue"))
            kb_local.ingest_pdf(str(test_pdf))
            metrics_local = kb_local.benchmark_retrieval(TEST_QUERIES, [[str(test_pdf)] * 4] * len(TEST_QUERIES), k=4)

            # Test OpenAI model
            console.print(Panel("Testing OpenAI model...", style="blue"))
            kb_openai.ingest_pdf(str(test_pdf))
            metrics_openai = kb_openai.benchmark_retrieval(TEST_QUERIES, [[str(test_pdf)] * 4] * len(TEST_QUERIES), k=4)

            # Populate metrics tables
            for metric, value in metrics_local.items():
                if isinstance(value, float):
                    local_metrics_table.add_row(metric.replace("_", " ").title(), f"{value:.3f}")

            for metric, value in metrics_openai.items():
                if isinstance(value, float):
                    openai_metrics_table.add_row(metric.replace("_", " ").title(), f"{value:.3f}")

            # Display results
            console.print("\n")
            layout["main"].split_row(Layout(local_metrics_table, ratio=1), Layout(openai_metrics_table, ratio=1))
            console.print(layout)

            # Performance assertions
            assert metrics_openai["avg_precision_at_k"] >= metrics_local["avg_precision_at_k"], (
                "OpenAI model should have better or equal precision"
            )
            assert metrics_openai["avg_recall_at_k"] >= metrics_local["avg_recall_at_k"], (
                "OpenAI model should have better or equal recall"
            )
            # Allow up to 5x slower for OpenAI model due to API latency
            assert metrics_openai["avg_retrieval_time"] < metrics_local["avg_retrieval_time"] * 5.0, (
                "OpenAI model is unexpectedly slow"
            )

            # Display improvement percentages
            improvements = Panel(
                f"""
                Precision Improvement: {((metrics_openai["avg_precision_at_k"] / metrics_local["avg_precision_at_k"]) - 1) * 100:.1f}%
                Recall Improvement: {((metrics_openai["avg_recall_at_k"] / metrics_local["avg_recall_at_k"]) - 1) * 100:.1f}%
                Speed Impact: {((metrics_openai["avg_retrieval_time"] / metrics_local["avg_retrieval_time"]) - 1) * 100:.1f}%
                """,
                title="Performance Improvements",
                style="green",
            )
            console.print(improvements)

        except Exception as e:
            console.print("[red]Error during embedding performance test[/red]")
            raise e
        finally:
            # Ensure cleanup happens even if test fails
            for kb in (kb_local, kb_openai):
                try:
                    if kb and kb.vector_store:
                        kb.reset_vector_store()
                except Exception as cleanup_error:
                    console.print(f"[red]Failed to cleanup vector store: {cleanup_error}[/red]")

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
                [kb.embeddings.embed_query(q) for q in TEST_QUERIES], k=test_config.k_retrieval
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
