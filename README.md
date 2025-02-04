# Reflective Agentic RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system implementing reflective agents and adaptive learning capabilities. This system combines state-of-the-art language models with intelligent document retrieval and dynamic strategy adjustment.

## Architecture Overview

The system is built on a modular architecture with four primary components:

```mermaid
---
config:
  look: neo
  theme: neo
---
flowchart TD
    %% Main Components
    Query[Query Input]
    Response[Generated Response]

    %% Query Processing Layer
    subgraph QueryProcessor[Query Processor]
        direction TB
        Intent[Intent Classification]
        Decomp[Query Decomposition]
        Intent --> Decomp
    end

    %% Knowledge Base Layer
    subgraph KnowledgeBase[Knowledge Base]
        direction TB
        DocProcess[Document Processing]
        TextSplit[Text Splitter<br/>Chunk Size: 500]
        VectorStore[Vector Store<br/>Chroma DB]
        Embed[Embeddings<br/>HuggingFace]
        DocProcess --> TextSplit
        TextSplit --> VectorStore
        VectorStore --- Embed
    end

    %% Agent Orchestration Layer
    subgraph AgentOrchestrator[Agent Orchestrator]
        direction LR
        subgraph Search[Search Agent]
            direction TB
            TopK[Top-k Retrieval]
            RelScore[Relevance Scoring]
            TopK --> RelScore
        end
        
        subgraph Reason[Reasoning Agent]
            direction TB
            Context[Context Analysis]
            CoT[Chain-of-Thought]
            Context --> CoT
        end
        
        subgraph ResponseAgent[Response Agent]
            direction TB
            Generate[Response Generation]
            Format[Format Control]
            Generate --> Format
        end
        
        Search --> Reason
        Reason --> ResponseAgent
    end

    %% Reflective Layer
    subgraph Reflective[Reflective Layer]
        direction TB
        Metrics[Performance Metrics]
        Strategy[Strategy Adjustment]
        Threshold[Performance Threshold<br/>0.8]
        History[Strategy History]
        
        Metrics --> Strategy
        Strategy --> Threshold
        Threshold --> History
    end

    %% External Services
    subgraph External[External Services]
        direction LR
        DeepSeek[DeepSeek LLM API]
        HuggingFace[HuggingFace Models]
    end

    %% Connections
    Query --> QueryProcessor
    QueryProcessor --> KnowledgeBase
    KnowledgeBase --> Search
    
    ResponseAgent --> Response
    
    %% External Service Connections
    DeepSeek -.-> ResponseAgent
    HuggingFace -.-> Embed
    
    %% Reflective Layer Connections
    ResponseAgent --> Metrics
    Reflective --> |Strategy Updates| Search
    Reflective --> |Performance Monitoring| Reason
    
    %% Tokyo Night Storm Theme Styling
    classDef processor fill:#24283b,stroke:#7aa2f7,stroke-width:2px,color:#c0caf5
    classDef knowledge fill:#24283b,stroke:#7dcfff,stroke-width:2px,color:#c0caf5
    classDef agent fill:#24283b,stroke:#bb9af7,stroke-width:2px,color:#c0caf5
    classDef reflective fill:#24283b,stroke:#9ece6a,stroke-width:2px,color:#c0caf5
    classDef external fill:#24283b,stroke:#e0af68,stroke-width:2px,color:#c0caf5
    classDef subcomponent fill:#1a1b26,stroke:#565f89,stroke-width:1px,color:#c0caf5
    
    class QueryProcessor processor
    class KnowledgeBase knowledge
    class AgentOrchestrator agent
    class Reflective reflective
    class External external
    class Intent,Decomp,DocProcess,TextSplit,VectorStore,Embed,TopK,RelScore,Context,CoT,Generate,Format,Metrics,Strategy,Threshold,History subcomponent
```

### Key Components

1. **Query Processor**
   - Intent classification using zero-shot learning
   - Query decomposition for complex questions
   - Optimized for both direct and analytical queries

2. **Knowledge Base**
   - Document processing with configurable chunk sizes (default: 500 tokens)
   - Chroma vector store integration for efficient retrieval
   - HuggingFace embeddings for semantic representation

3. **Agent Orchestrator**
   - Search Agent: Handles document retrieval with dynamic top-k selection
   - Reasoning Agent: Implements chain-of-thought prompting
   - Response Agent: Manages response generation and formatting

4. **Reflective Layer**
   - Continuous performance monitoring
   - Dynamic strategy adjustment based on performance metrics
   - Adaptive thresholds for optimization (default: 0.8)

## Command Line Interface

The system includes a comprehensive CLI for easy interaction:

```mermaid
---
config:
  look: neo
  theme: neo
---
flowchart TD
    %% Main Menu
    Start([Start]) --> MainMenu{Main Menu}
    MainMenu --> |Select| LoadPDFs[Load PDFs]
    MainMenu --> |Select| QuerySystem[Query System]
    MainMenu --> |Select| ViewMetrics[View Performance Metrics]
    MainMenu --> |Select| Exit([Exit])

    %% PDF Loading Menu
    LoadPDFs --> PDFMenu{PDF Loading Menu}
    PDFMenu --> |Option 1| SinglePDF[Load Single PDF]
    PDFMenu --> |Option 2| MultiplePDFs[Load Multiple PDFs]
    PDFMenu --> |Option 3| ViewLoaded[View Loaded PDFs]
    PDFMenu --> |Option 4| BackToMain1[Back to Main Menu]

    %% Single PDF Flow
    SinglePDF --> InputPath[Enter PDF Path]
    InputPath --> ValidatePath{Path Valid?}
    ValidatePath --> |Yes| LoadProgress[Show Loading Progress]
    ValidatePath --> |No| RetryPrompt{Retry?}
    RetryPrompt --> |Yes| InputPath
    RetryPrompt --> |No| PDFMenu
    LoadProgress --> UpdateLoadedPDFs[Update Loaded PDFs Set]

    %% Multiple PDF Flow
    MultiplePDFs --> InputDir[Enter Directory Path]
    InputDir --> ValidateDir{Directory Valid?}
    ValidateDir --> |Yes| ListPDFs[List Available PDFs]
    ValidateDir --> |No| RetryDir{Retry?}
    RetryDir --> |Yes| InputDir
    RetryDir --> |No| PDFMenu
    ListPDFs --> SelectPDFs[Select PDFs to Load]
    SelectPDFs --> BatchProgress[Show Batch Progress]
    BatchProgress --> UpdateLoadedPDFs

    %% Query Menu
    QuerySystem --> QueryMenu{Query Menu}
    QueryMenu --> |Option 1| AskQuestion[Ask Question]
    QueryMenu --> |Option 2| ViewPrevious[View Previous Questions]
    QueryMenu --> |Option 3| BackToMain2[Back to Main Menu]

    %% Performance Metrics
    ViewMetrics --> LoadMetrics[Load Performance Metrics]
    LoadMetrics --> MetricsExist{Metrics Available?}
    MetricsExist --> |Yes| ShowMetrics[Display Metrics]
    MetricsExist --> |No| NoMetrics[Show No Metrics Message]

    %% Tokyo Night Storm Theme Styling
    classDef menu fill:#24283b,stroke:#7aa2f7,stroke-width:2px,color:#c0caf5
    classDef process fill:#1a1b26,stroke:#7dcfff,stroke-width:1px,color:#c0caf5
    classDef decision fill:#24283b,stroke:#bb9af7,stroke-width:2px,color:#c0caf5
    classDef terminator fill:#24283b,stroke:#9ece6a,stroke-width:2px,color:#c0caf5
    
    class MainMenu,PDFMenu,QueryMenu menu
    class LoadProgress,ProcessQuestion,ShowMetrics,UpdateLoadedPDFs,ListPDFs,SelectPDFs,BatchProgress,AskQuestion,ViewPrevious,LoadMetrics,ShowMetrics,NoMetrics process
    class ValidatePath,ValidateDir,MetricsExist,RetryPrompt,RetryDir decision
    class Start,Exit terminator
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/reflective-rag.git
cd reflective-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- Python 3.8+
- DeepSeek API key
- HuggingFace Transformers
- LangChain
- ChromaDB
- Rich (for CLI interface)
- Questionary (for interactive prompts)

## Configuration

1. Set up environment variables:
```bash
export DEEPSEEK_API_KEY="your-api-key"
```

2. Configure RAG settings in `config.py`:
```python
RAG_CONFIG = {
    "chunk_size": 500,
    "chunk_overlap": 50,
    "k_retrieval": 4,
    "performance_threshold": 0.8
}
```

## Usage

### Command Line Interface

1. Start the CLI:
```bash
python cli.py
```

2. Load Documents:
   - Single PDF: Enter path to PDF file
   - Multiple PDFs: Select from directory
   - View loaded documents

3. Query the System:
   - Ask questions about loaded documents
   - View previous questions and performance
   - Monitor system metrics

### Python API

```python
from rag_system import RAGConfig, ReflectiveRAG

# Initialize configuration
config = RAGConfig(
    llm_api_key="your-api-key",
    llm_api_base_url="https://api.deepseek.com/v1"
)

# Create RAG instance
rag = ReflectiveRAG(config)

# Load documents
rag.knowledge_base.ingest_pdf("path/to/document.pdf")

# Query the system
response = rag.answer_query("What are the key points in the document?")
```

## Performance Metrics

The system tracks several key metrics:

1. **Relevance Scores**
   - Document retrieval accuracy
   - Context utilization

2. **Coherence Metrics**
   - Response consistency
   - Context adherence

3. **System Performance**
   - Response time
   - Resource utilization

## Advanced Features

### Adaptive Retrieval
- Dynamic k selection based on query complexity
- Context length optimization
- Relevance threshold adjustment

### Reflective Learning
- Performance history tracking
- Strategy adaptation
- Query optimization

### Error Handling
- Robust PDF processing
- Invalid input management
- API failure recovery

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push branch (`git push origin feature/improvement`)
5. Open Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details

## Acknowledgments

- DeepSeek API for LLM capabilities
- HuggingFace for embeddings and models
- LangChain community for core RAG components

