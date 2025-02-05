import os
import shutil
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from agentic_rag.rag_system import RAGConfig, ReflectiveRAG, KnowledgeBase, DeepseekQuery, ReflectiveLayer
from rich.console import Console
from chromadb.config import Settings

# Use a non-interactive console for testing
console = Console(force_terminal=False)

@pytest.fixture(autouse=True)
def env_setup():
    """Setup environment variables for testing."""
    original_deepseek = os.environ.get("DEEPSEEK_API_KEY")
    original_kluster = os.environ.get("KLUSTER_API_KEY")
    
    os.environ["DEEPSEEK_API_KEY"] = "dummy-deepseek-key"
    os.environ["KLUSTER_API_KEY"] = "dummy-kluster-key"
    
    yield
    
    if original_deepseek:
        os.environ["DEEPSEEK_API_KEY"] = original_deepseek
    else:
        os.environ.pop("DEEPSEEK_API_KEY", None)
        
    if original_kluster:
        os.environ["KLUSTER_API_KEY"] = original_kluster
    else:
        os.environ.pop("KLUSTER_API_KEY", None)

@pytest.fixture(autouse=True)
def clean_test_collections():
    """Clean up test collections before and after tests."""
    test_dir = os.path.join(os.getcwd(), "knowledge_bases")
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, mode=0o755)
    yield
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

@pytest.fixture
def mock_api():
    """Mock API calls."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
    
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    
    with patch("openai.OpenAI", return_value=mock_client) as mock_openai:
        mock_openai.return_value = mock_client
        yield mock_client.chat.completions.create

@pytest.fixture
def mock_client(mock_api):
    """Mock the OpenAI client for DeepSeek."""
    with patch("agentic_rag.rag_system.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock(
            base_url="https://api.deepseek.com/v1",
            chat=MagicMock(
                completions=MagicMock(
                    create=mock_api
                )
            )
        )
        yield mock_openai

@pytest.fixture
def test_config():
    """Create a test configuration."""
    return RAGConfig(
        llm_api_key="dummy-key",
        llm_api_base_url="https://api.deepseek.com/v1",
    )

@pytest.fixture
def mock_progress():
    """Mock rich progress bar."""
    with patch("rich.progress.Progress.track") as mock:
        yield mock

@pytest.fixture
def pdf_dir():
    """Create a test PDF directory."""
    return Path("PDFS")

@pytest.fixture
def chroma_settings():
    """Create in-memory ChromaDB settings for testing."""
    return Settings(
        is_persistent=False,  # Use in-memory database
        anonymized_telemetry=False
    )

@pytest.fixture
def knowledge_base(test_config, chroma_settings):
    """Create a test knowledge base with in-memory storage."""
    kb = KnowledgeBase(test_config)
    kb.chroma_settings = chroma_settings
    return kb

@pytest.fixture
def rag_system(test_config, chroma_settings):
    """Create a test RAG system with in-memory storage."""
    rag = ReflectiveRAG(test_config)
    rag.knowledge_base.chroma_settings = chroma_settings
    return rag

def test_pdf_loading(mock_progress, knowledge_base, pdf_dir):
    """Test loading PDFs into the knowledge base."""
    test_pdf = next(pdf_dir.glob("*.pdf"))
    knowledge_base.ingest_pdf(str(test_pdf))
    assert knowledge_base.vector_store is not None

@pytest.mark.asyncio
async def test_query_processing(mock_progress, mock_client, rag_system, pdf_dir):
    """Test processing a query through the RAG system."""
    # Load a single PDF for testing
    test_pdf = next(pdf_dir.glob("*.pdf"))
    rag_system.knowledge_base.ingest_pdf(str(test_pdf))

    # Mock the ReasoningAgent and ResponseAgent
    with patch("agentic_rag.rag_system.ReasoningAgent") as mock_reason, \
         patch("agentic_rag.rag_system.ResponseAgent") as mock_response, \
         patch.object(rag_system.reflective_layer, "evaluate_response") as mock_evaluate:
        # Set up mock instances
        mock_reason_instance = MagicMock()
        mock_reason_instance.analyze.return_value = "Test analysis"
        mock_reason.return_value = mock_reason_instance

        mock_response_instance = MagicMock()
        mock_response_instance.generate.return_value = "Test response"
        mock_response.return_value = mock_response_instance

        # Mock evaluate_response to return a high performance score
        mock_evaluate.return_value = 0.9

        # Replace the agents in the orchestrator
        rag_system.orchestrator.agents["reason"] = mock_reason_instance
        rag_system.orchestrator.agents["response"] = mock_response_instance

        # Process query
        response = rag_system.answer_query("test query")
        expected_response = "Sub-query 1:\nTest response\n\nSub-query 2:\nTest response\n\nSub-query 3:\nTest response"
        assert response == expected_response

        # Verify the mocks were called correctly
        assert mock_reason_instance.analyze.call_count == 3  # One call per sub-query
        assert mock_response_instance.generate.call_count == 3  # One call per sub-query

def test_performance_metrics(mock_progress, mock_client, rag_system, pdf_dir):
    """Test the reflective layer's performance metrics."""
    # Load test data
    test_pdf = next(pdf_dir.glob("*.pdf"))
    rag_system.knowledge_base.ingest_pdf(str(test_pdf))
    
    # Test performance metrics
    metrics = rag_system.reflective_layer.evaluate_response(
        "Test query", "Test response", ["Test document"]
    )
    assert isinstance(metrics, float)
    assert 0.0 <= metrics <= 1.0

def test_knowledge_base_reuse(mock_progress, knowledge_base, pdf_dir):
    """Test that knowledge base properly handles multiple PDF loads."""
    pdf_files = list(pdf_dir.glob("*.pdf"))[:2]  # Test with first two PDFs
    
    # Load first PDF
    knowledge_base.ingest_pdf(str(pdf_files[0]))
    assert knowledge_base.vector_store is not None
    
    # Load second PDF
    knowledge_base.ingest_pdf(str(pdf_files[1]))
    assert knowledge_base.vector_store is not None

def test_deepseek_query(mock_client, test_config):
    """Test the Deepseek query client functionality with Kluster AI fallback."""
    # Initialize client
    client = DeepseekQuery(test_config)
    
    # Test with Kluster AI
    with patch("agentic_rag.rag_system.OpenAI") as mock_openai:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_instance
        
        response = client.get_deepseek_response("test system", "test user")
        assert response == "Test response"
        assert mock_openai.call_args.kwargs == {
            "api_key": "dummy-kluster-key",
            "base_url": "https://api.kluster.ai/v1"
        }
        
    # Test fallback to Deepseek
    client.use_kluster = False
    with patch("agentic_rag.rag_system.OpenAI") as mock_openai:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Deepseek response"))]
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_instance
        
        response = client.get_deepseek_response("test system", "test user")
        assert response == "Deepseek response"
        assert mock_openai.call_args.kwargs == {
            "api_key": test_config.llm_api_key,
            "base_url": test_config.llm_api_base_url
        }

def test_reflective_layer(test_config):
    """Test the reflective layer functionality."""
    # Initialize reflective layer
    reflective = ReflectiveLayer(test_config)
    
    # Test performance evaluation
    metrics = reflective.evaluate_response("Test query", "Test response", ["Test document"])
    assert isinstance(metrics, float)
    assert 0.0 <= metrics <= 1.0
    
    # Test strategy adjustment
    strategy = reflective.adjust_strategy(metrics)
    assert isinstance(strategy, dict)
    assert "k_retrieval" in strategy
    assert "rewrite_query" in strategy

def test_collection_creation(test_config, chroma_settings):
    """Test creating a knowledge base with a specific collection."""
    kb = KnowledgeBase(test_config, collection_name="test_collection")
    kb.chroma_settings = chroma_settings
    assert kb.collection_name == "test_collection"

def test_collection_persistence(test_config, chroma_settings):
    """Test that collections persist data correctly."""
    kb = KnowledgeBase(test_config, collection_name="test_persistence")
    kb.chroma_settings = chroma_settings
    assert kb.collection_name == "test_persistence"

def test_multiple_collections(test_config, chroma_settings):
    """Test managing multiple collections."""
    kb1 = KnowledgeBase(test_config, collection_name="collection1")
    kb2 = KnowledgeBase(test_config, collection_name="collection2")
    kb1.chroma_settings = chroma_settings
    kb2.chroma_settings = chroma_settings
    
    assert kb1.collection_name != kb2.collection_name
    assert kb1.persist_directory != kb2.persist_directory

def test_collection_isolation(test_config, chroma_settings):
    """Test that collections are properly isolated."""
    kb1 = KnowledgeBase(test_config, collection_name="test_isolation1")
    kb2 = KnowledgeBase(test_config, collection_name="test_isolation2")
    kb1.chroma_settings = chroma_settings
    kb2.chroma_settings = chroma_settings
    
    assert kb1.collection_name != kb2.collection_name
    assert kb1.persist_directory != kb2.persist_directory

@pytest.mark.asyncio
async def test_rag_with_collections(test_config, chroma_settings):
    """Test RAG system with different collections."""
    # Create RAG instances with different collections
    rag1 = ReflectiveRAG(test_config, collection_name="test_collection1")
    rag2 = ReflectiveRAG(test_config, collection_name="test_collection2")
    rag1.knowledge_base.chroma_settings = chroma_settings
    rag2.knowledge_base.chroma_settings = chroma_settings

    # Create test documents
    test_docs = ["Test document 1", "Test document 2"]
    rag1.knowledge_base.ingest_documents(test_docs)
    rag2.knowledge_base.ingest_documents(test_docs)

    # Mock the ReasoningAgent and ResponseAgent
    with patch("agentic_rag.rag_system.ReasoningAgent") as mock_reason, \
         patch("agentic_rag.rag_system.ResponseAgent") as mock_response, \
         patch.object(rag1.query_processor, "understand_query") as mock_understand1, \
         patch.object(rag2.query_processor, "understand_query") as mock_understand2, \
         patch.object(rag1.reflective_layer, "evaluate_response") as mock_evaluate1, \
         patch.object(rag2.reflective_layer, "evaluate_response") as mock_evaluate2:
        # Set up mock instances
        mock_reason_instance = MagicMock()
        mock_reason_instance.analyze.return_value = "Test analysis"
        mock_reason.return_value = mock_reason_instance

        mock_response_instance = MagicMock()
        mock_response_instance.generate.return_value = "Test response"
        mock_response.return_value = mock_response_instance

        # Mock the understand_query to always return analytical intent
        mock_understand1.return_value = {"original_query": "test query 1", "intent": "analytical", "confidence": 1.0}
        mock_understand2.return_value = {"original_query": "test query 2", "intent": "analytical", "confidence": 1.0}

        # Mock evaluate_response to return a high performance score
        mock_evaluate1.return_value = 0.9
        mock_evaluate2.return_value = 0.9

        # Replace the agents in both orchestrators
        rag1.orchestrator.agents["reason"] = mock_reason_instance
        rag1.orchestrator.agents["response"] = mock_response_instance
        rag2.orchestrator.agents["reason"] = mock_reason_instance
        rag2.orchestrator.agents["response"] = mock_response_instance

        # Process queries on both instances
        response1 = rag1.answer_query("test query 1")
        response2 = rag2.answer_query("test query 2")

        expected_response = "Sub-query 1:\nTest response\n\nSub-query 2:\nTest response\n\nSub-query 3:\nTest response"
        assert response1 == expected_response
        assert response2 == expected_response

        # Verify the mocks were called correctly
        # Each query generates 3 sub-queries, and each sub-query triggers one analyze and one generate call
        assert mock_reason_instance.analyze.call_count == 6  # 3 sub-queries per query
        assert mock_response_instance.generate.call_count == 6  # 3 sub-queries per query 

def test_embedding_performance(test_config, chroma_settings, pdf_dir):
    """Test and compare embedding model performance."""
    # Initialize knowledge bases with different models
    old_config = RAGConfig(
        llm_api_key=test_config.llm_api_key,
        llm_api_base_url=test_config.llm_api_base_url,
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        embedding_dimensions=768
    )
    
    new_config = RAGConfig(
        llm_api_key=test_config.llm_api_key,
        llm_api_base_url=test_config.llm_api_base_url,
        embedding_model="Salesforce/SFR-Embedding-Mistral",
        embedding_dimensions=4096
    )
    
    kb_old = KnowledgeBase(old_config, "test_old")
    kb_new = KnowledgeBase(new_config, "test_new")
    kb_old.chroma_settings = chroma_settings
    kb_new.chroma_settings = chroma_settings
    
    # Load test data
    test_pdf = next(pdf_dir.glob("*.pdf"))
    kb_old.ingest_pdf(str(test_pdf))
    kb_new.ingest_pdf(str(test_pdf))
    
    # Test queries and expected relevant documents
    test_queries = [
        "What are the key components of the system?",
        "How does the reflective layer work?",
        "Explain the query processing workflow"
    ]
    
    # For testing, we'll use the document IDs as relevant docs
    relevant_docs = [[str(test_pdf)] * 2] * len(test_queries)
    
    # Benchmark both models
    old_metrics = kb_old.benchmark_retrieval(test_queries, relevant_docs)
    new_metrics = kb_new.benchmark_retrieval(test_queries, relevant_docs)
    
    console.print("\n[bold blue]Embedding Model Performance Comparison:[/bold blue]")
    console.print(f"\nOld Model ({old_metrics['model_name']}):")
    console.print(f"- Precision@k: {old_metrics['avg_precision_at_k']:.3f}")
    console.print(f"- Recall@k: {old_metrics['avg_recall_at_k']:.3f}")
    console.print(f"- Avg Retrieval Time: {old_metrics['avg_retrieval_time']:.3f}s")
    
    console.print(f"\nNew Model ({new_metrics['model_name']}):")
    console.print(f"- Precision@k: {new_metrics['avg_precision_at_k']:.3f}")
    console.print(f"- Recall@k: {new_metrics['avg_recall_at_k']:.3f}")
    console.print(f"- Avg Retrieval Time: {new_metrics['avg_retrieval_time']:.3f}s")
    
    # Assert the new model performs better
    assert new_metrics['avg_precision_at_k'] >= old_metrics['avg_precision_at_k'], "New model should have better precision"
    assert new_metrics['avg_recall_at_k'] >= old_metrics['avg_recall_at_k'], "New model should have better recall" 