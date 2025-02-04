import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from agentic_rag.rag_system import RAGConfig, ReflectiveRAG, KnowledgeBase
from rich.console import Console

# Use a non-interactive console for testing
console = Console(force_terminal=False)

@pytest.fixture(autouse=True)
def env_setup():
    """Setup environment variables for testing."""
    original_key = os.environ.get("DEEPSEEK_API_KEY")
    os.environ["DEEPSEEK_API_KEY"] = "dummy-key"
    yield
    if original_key:
        os.environ["DEEPSEEK_API_KEY"] = original_key
    else:
        os.environ.pop("DEEPSEEK_API_KEY", None)

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
    """Create test config with DeepSeek settings."""
    return RAGConfig(
        llm_api_key="dummy-key",
        llm_api_base_url="https://api.deepseek.com/v1",
    )

@pytest.fixture
def pdf_dir():
    return Path("PDFS")

@pytest.fixture
def rag_system(test_config):
    return ReflectiveRAG(test_config)

@pytest.fixture
def knowledge_base(test_config):
    return KnowledgeBase(test_config)

def test_pdf_loading(pdf_dir):
    """Test that PDFs can be loaded from the directory."""
    assert pdf_dir.exists(), "PDF directory not found"
    pdf_files = list(pdf_dir.glob("*.pdf"))
    assert len(pdf_files) > 0, "No PDF files found in directory"
    console.print(f"[green]Found {len(pdf_files)} PDF files[/green]")

@patch("rich.progress.Progress.track")
def test_knowledge_base_creation(mock_progress, knowledge_base, pdf_dir):
    """Test knowledge base creation and document ingestion."""
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    # Test loading each PDF
    for pdf_file in pdf_files:
        try:
            knowledge_base.ingest_pdf(str(pdf_file))
            console.print(f"[green]Successfully loaded {pdf_file.name}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to load {pdf_file.name}: {str(e)}[/red]")
            raise

@patch("rich.progress.Progress.track")
def test_query_processing(mock_progress, mock_client, rag_system, pdf_dir):
    """Test the query processing pipeline."""
    # Load a single PDF for testing
    test_pdf = next(pdf_dir.glob("*.pdf"))
    rag_system.knowledge_base.ingest_pdf(str(test_pdf))
    
    # Test simple query
    query = "What is the main topic of this paper?"
    response = rag_system.answer_query(query)
    
    # The response will include sub-queries due to our query processor
    expected_response = "Sub-query 1:\nTest response\n\nSub-query 2:\nTest response\n\nSub-query 3:\nTest response"
    assert response == expected_response
    console.print(f"[green]Query response received: {response[:100]}...[/green]")

@patch("rich.progress.Progress.track")
def test_performance_metrics(mock_progress, mock_client, rag_system, pdf_dir):
    """Test the reflective layer's performance metrics."""
    # Load test data
    test_pdf = next(pdf_dir.glob("*.pdf"))
    rag_system.knowledge_base.ingest_pdf(str(test_pdf))
    
    # Process a query and check metrics
    query = "Summarize the methodology section."
    response = rag_system.answer_query(query)
    metrics = rag_system.reflective_layer.performance_metrics
    
    assert len(metrics) > 0, "No performance metrics recorded"
    assert all(0.0 <= m["relevance"] <= 1.0 for m in metrics), "Invalid relevance scores"
    assert all(0.0 <= m["coherence"] <= 1.0 for m in metrics), "Invalid coherence scores"

@patch("rich.progress.Progress.track")
def test_knowledge_base_reuse(mock_progress, knowledge_base, pdf_dir):
    """Test that knowledge base properly handles multiple PDF loads."""
    pdf_files = list(pdf_dir.glob("*.pdf"))[:2]  # Test with first two PDFs
    
    # Load first PDF
    knowledge_base.ingest_pdf(str(pdf_files[0]))
    assert knowledge_base.vector_store is not None
    
    # Load second PDF
    knowledge_base.ingest_pdf(str(pdf_files[1]))
    # Vector store should still exist and contain both documents
    assert knowledge_base.vector_store is not None
    
    # Test retrieval
    result = knowledge_base.retrieve("test query")
    assert len(result) > 0, "Should retrieve documents from vector store"

def test_deepseek_query(mock_client, test_config):
    """Test the Deepseek query client functionality."""
    from agentic_rag.rag_system import DeepseekQuery
    
    # Initialize Deepseek client
    client = DeepseekQuery(test_config)
    
    # Test simple query
    system_prompt = "You are a helpful assistant"
    user_prompt = "What is photosynthesis?"
    
    response = client.get_deepseek_response(system_prompt, user_prompt)
    assert response == "Test response"
    
    # Verify the mock was called with correct parameters
    mock_client.return_value.chat.completions.create.assert_called_once_with(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1024,
        temperature=0.7,
        stream=False
    )

def test_reflective_layer(test_config):
    """Test the reflective layer functionality."""
    from agentic_rag.rag_system import ReflectiveLayer
    
    # Initialize reflective layer
    reflective = ReflectiveLayer(test_config)
    
    # Test evaluation
    query = "Test query"
    response = "Test response"
    retrieved_docs = ["Doc 1", "Doc 2"]
    
    # Test response evaluation
    score = reflective.evaluate_response(query, response, retrieved_docs)
    assert 0.0 <= score <= 1.0, "Score should be between 0 and 1"
    
    # Test metrics storage
    assert len(reflective.performance_metrics) == 1
    metric = reflective.performance_metrics[0]
    assert metric["query"] == query
    assert 0.0 <= metric["relevance"] <= 1.0
    assert 0.0 <= metric["coherence"] <= 1.0
    
    # Test strategy adjustment
    strategy = reflective.adjust_strategy(score)
    assert "k_retrieval" in strategy
    assert "rewrite_query" in strategy
    assert isinstance(strategy["k_retrieval"], int)
    assert isinstance(strategy["rewrite_query"], bool)
    
    # Verify strategy history
    assert len(reflective.strategy_history) == 1
    assert reflective.strategy_history[0] == strategy 