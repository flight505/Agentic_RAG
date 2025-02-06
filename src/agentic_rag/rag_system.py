import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from chromadb import Settings as ChromaSettings
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console

# Initialize console for rich output
console = Console()

# Device configuration
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
console.print(f"[green]Using device: {DEVICE}[/green]")

# Core dependencies with error handling
try:
    from datasets import load_dataset
    from transformers import pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    console.print("[red]Warning: transformers package not available. Some features will be disabled.[/red]")
    TRANSFORMERS_AVAILABLE = False

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import OpenAIEmbeddings
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    console.print(
        "[red]Warning: embedding packages not available. Vector store functionality will be disabled.[/red]"
    )
    EMBEDDINGS_AVAILABLE = False

# Add constants at the top of the file after imports
PERFORMANCE_THRESHOLD = 0.8  # Threshold for strategy adjustment

class RAGConfig(BaseModel):
    """Configuration for the RAG system."""
    model_config = ConfigDict(validate_default=True)  # Validate defaults but allow modifications
    
    llm_api_key: str = Field(description="Your LLM API key")
    llm_api_base_url: str = Field(description="Your LLM API base URL")
    llm_api_version: str = Field(default="2025-02-01", description="API version if required")
    llm_deployment_name: str | None = Field(default=None, description="For Azure deployment")

    # Embedding Configuration
    embedding_provider: str = Field(
        default="openai",
        description="'openai' or 'local'",
        pattern="^(openai|local)$"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-large",
        description="OpenAI's best model"
    )
    openai_dimensions: int = Field(
        default=3072,
        description="Can be reduced for performance/cost tradeoff"
    )
    local_embedding_model: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        description="Fallback local model"
    )
    local_dimensions: int = Field(
        default=768,
        description="Local model dimensions"
    )

    # RAG Configuration
    chunk_size: int = Field(default=500, gt=0, description="Size of text chunks")
    chunk_overlap: int = Field(default=50, ge=0, description="Overlap between chunks")
    k_retrieval: int = Field(default=4, gt=0, description="Number of documents to retrieve")

    @property
    def is_valid(self) -> bool:
        """Check if the configuration is valid."""
        return bool(self.llm_api_key and self.llm_api_base_url and 
                   self.embedding_provider in ["openai", "local"])

    def model_post_init(self, __context: Any) -> None:
        """Validate configuration after initialization."""
        if not self.is_valid:
            raise ValueError("Invalid configuration: API key, base URL required and valid embedding provider")


class DeepseekQuery:
    def __init__(self, config: RAGConfig, client: OpenAI | None = None):
        self.config = config
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {config.llm_api_key}"}
        self.kluster_key = os.getenv("KLUSTER_API_KEY")
        self.use_kluster = bool(self.kluster_key)
        self._client = client

    def get_deepseek_response(self, system_content: str, user_content: str) -> str:
        """Get response from DeepSeek or Kluster AI with proper fallback."""
        try:
            client = self._client
            if not client:
                client = OpenAI(
                    api_key=self.kluster_key if self.use_kluster else self.config.llm_api_key,
                    base_url="https://api.kluster.ai/v1" if self.use_kluster else self.config.llm_api_base_url
                )
            
            response = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-R1" if self.use_kluster else "deepseek-chat",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=1024,
                temperature=0.7,
                stream=False,
            )
            
            if response and response.choices:
                return response.choices[0].message.content
            raise Exception("Error: No response content received")
                
        except Exception as e:
            console.print(f"[red]Error in LLM query: {e!s}[/red]")
            # Fallback to Kluster if DeepSeek fails
            if not self.use_kluster and self.kluster_key:
                self.use_kluster = True
                return self.get_deepseek_response(system_content, user_content)
            raise


class QueryProcessor:
    def __init__(self):
        if not TRANSFORMERS_AVAILABLE:
            console.print("[yellow]Warning: Running without transformers. Using basic query processing.[/yellow]")
            self.intent_classifier = None
        else:
            self.intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def understand_query(self, query: str) -> dict[str, Any]:
        if not self.intent_classifier:
            return {
                "original_query": query,
                "intent": "factual",  # Default to factual when no classifier
                "confidence": 1.0,
            }

        try:
            intents = self.intent_classifier(query, candidate_labels=["factual", "analytical", "procedural"])
            return {"original_query": query, "intent": intents["labels"][0], "confidence": intents["scores"][0]}
        except Exception as e:
            console.print(f"[red]Error in query classification: {e!s}[/red]")
            return {"original_query": query, "intent": "factual", "confidence": 1.0}

    def decompose_query(self, query_info: dict[str, Any]) -> list[str]:
        if query_info["intent"] == "analytical":
            sub_queries = [
                f"What are the key facts about {query_info['original_query']}?",
                f"What are the relationships between different aspects of {query_info['original_query']}?",
                f"What are the implications of {query_info['original_query']}?",
            ]
        else:
            sub_queries = [query_info["original_query"]]
        return sub_queries


class KnowledgeBase:
    def __init__(self, config: RAGConfig, collection_name: str = "default"):
        self.config = config
        self.collection_name = collection_name
        self.persist_directory = os.path.join(os.getcwd(), "knowledge_bases", collection_name)
        
        if not EMBEDDINGS_AVAILABLE:
            console.print(
                "[red]Error: Embedding packages not available. Vector store functionality will be disabled.[/red]"
            )
            self.embeddings = None
        else:
            try:
                console.print("[cyan]Initializing embeddings...[/cyan]")
                if config.embedding_provider == "openai":
                    console.print("[bold green]Setting up OpenAI embeddings...")
                    self.embeddings = OpenAIEmbeddings(
                        model=config.openai_embedding_model,
                        dimensions=config.openai_dimensions,
                        client=OpenAI()
                    )
                    console.print(
                        f"[green]Successfully initialized OpenAI embeddings: "
                        f"{config.openai_embedding_model} ({config.openai_dimensions} dimensions)[/green]"
                    )
                else:
                    console.print("[bold green]Loading local embedding model...")
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name=config.local_embedding_model,
                        model_kwargs={"device": DEVICE},
                        encode_kwargs={"normalize_embeddings": True}
                    )
                    console.print(
                        f"[green]Successfully loaded local model: "
                        f"{config.local_embedding_model} on {DEVICE}[/green]"
                    )
            except Exception as e:
                console.print(f"[red]Error initializing embeddings: {e!s}[/red]")
                if config.embedding_provider == "openai":
                    console.print("[yellow]Falling back to local embeddings...[/yellow]")
                    try:
                        self.embeddings = HuggingFaceEmbeddings(
                            model_name=config.local_embedding_model,
                            model_kwargs={"device": DEVICE},
                            encode_kwargs={"normalize_embeddings": True}
                        )
                        console.print(
                            f"[green]Successfully loaded fallback model: "
                            f"{config.local_embedding_model}[/green]"
                        )
                    except Exception as e2:
                        console.print(f"[red]Error loading fallback embeddings: {e2!s}[/red]")
                        self.embeddings = None
                else:
                    self.embeddings = None

        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
        # Create persist directory with proper permissions
        os.makedirs(self.persist_directory, mode=0o755, exist_ok=True)
        
        # Initialize ChromaDB settings with new format
        self.chroma_settings = ChromaSettings(
            is_persistent=True,
            persist_directory=self.persist_directory,
            anonymized_telemetry=False
        )
        
        # Try to load existing vector store
        try:
            if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name,
                    client_settings=self.chroma_settings
                )
                console.print(f"[green]Loaded existing knowledge base: {self.collection_name}[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not load existing knowledge base: {e}[/yellow]")

    def convert_pdf_to_text(self, pdf_path: str) -> str:
        """Convert a PDF file to text using the Marker library.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            Extracted text from the PDF.
            
        Raises:
            Exception: If conversion fails or produces empty text.
        """
        try:
            import logging

            from marker.config.parser import ConfigParser
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict

            # Configure handler to capture progress
            class RichProgressHandler(logging.Handler):
                def __init__(self, console):
                    super().__init__()
                    self.console = console
                    
                def emit(self, record):
                    msg = record.getMessage()
                    
                    # Handle model loading messages
                    if "Loaded" in msg and "model" in msg:
                        self.console.print(msg, style="green")
                        return
                    
                    # Handle progress messages
                    for step in ['layout', 'texify', 'equations', 'tables']:
                        if step in msg.lower():
                            self.console.print(f"Processing {step}... {msg}", style="cyan")

            # Set up logging with our custom handler
            handler = RichProgressHandler(console)
            logger = logging.getLogger('marker')
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)

            config = {
                "output_format": "markdown",
                "use_llm": False,
                "force_ocr": False,
            }
            
            config_parser = ConfigParser(config)
            converter = PdfConverter(
                config=config_parser.generate_config_dict(),
                artifact_dict=create_model_dict(),
            )
            
            # Run conversion with status updates through logging
            with console.status(f"[cyan]Converting {Path(pdf_path).name}...", spinner="dots") as status:
                text = converter(pdf_path).markdown
                
                if not text or len(text.strip()) == 0:
                    raise ValueError("Conversion produced empty text")
            
            # Clean up logging handler
            logger.removeHandler(handler)
            return text
            
        except Exception as e:
            raise Exception(f"Error converting PDF '{Path(pdf_path).name}': {e!s}") from e

    def ingest_pdf(self, pdf_path: str) -> None:
        """Ingest a PDF file into the knowledge base.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Raises:
            ValueError: If embeddings are not available.
            Exception: If PDF processing fails.
        """
        if not self.embeddings:
            raise ValueError("Embeddings not available. Cannot ingest PDF.")
        
        try:
            # Convert PDF to text first
            text = self.convert_pdf_to_text(pdf_path)
            
            # Create a Document and split into chunks
            doc = Document(
                page_content=text,
                metadata={
                    "source": pdf_path,
                    "filename": Path(pdf_path).name,
                    "collection": self.collection_name
                }
            )
            
            split_docs = self.text_splitter.split_documents([doc])
            
            # Initialize or update vector store
            if self.vector_store is None:
                self.vector_store = Chroma.from_documents(
                    documents=split_docs,
                    embedding=self.embeddings,
                    collection_name=self.collection_name,
                    client_settings=self.chroma_settings
                )
            else:
                self.vector_store.add_documents(split_docs)
            
        except Exception as e:
            raise Exception(f"Error processing PDF {Path(pdf_path).name}: {e!s}") from e

    def ingest_documents(self, documents: list[str], source_type: str = "text"):
        if not self.embeddings:
            raise ValueError("Embeddings not available. Cannot ingest documents.")

        if source_type == "text":
            docs = []
            with console.status("[cyan]Processing text documents...") as status:
                for i, doc in enumerate(documents, 1):
                    try:
                        status.update(f"[cyan]Processing document {i}/{len(documents)}...")
                        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as temp_file:
                            temp_file.write(doc)
                            temp_file_path = temp_file.name
                        docs.extend(TextLoader(temp_file_path, encoding="utf-8").load())
                        os.remove(temp_file_path)
                    except Exception as e:
                        console.print(f"[red]Error processing document {i}: {e!s}[/red]")
        elif source_type == "pdf":
            docs = []
            with console.status("[cyan]Processing PDF documents...") as status:
                for i, doc in enumerate(documents, 1):
                    try:
                        status.update(f"[cyan]Processing PDF {i}/{len(documents)}: {Path(doc).name}...")
                        docs.extend(PyPDFLoader(doc).load())
                    except Exception as e:
                        console.print(f"[red]Error loading PDF {doc}: {e!s}[/red]")

        if not docs:
            raise ValueError("No documents were successfully processed")

        try:
            with console.status("[cyan]Splitting documents...") as status:
                split_docs = self.text_splitter.split_documents(docs)
                status.update("[cyan]Creating vector store...")
                if self.vector_store is None:
                    self.vector_store = Chroma.from_documents(
                        documents=split_docs,
                        embedding=self.embeddings,
                        collection_name=self.collection_name,
                        client_settings=self.chroma_settings
                    )
                else:
                    self.vector_store.add_documents(split_docs)
                console.print(f"[green]Successfully processed {len(split_docs)} document chunks[/green]")
        except Exception as e:
            console.print(f"[red]Error creating vector store: {e!s}[/red]")
            raise

    def retrieve(self, query: str) -> list[str]:
        if not self.vector_store:
            raise ValueError("No documents ingested yet!")
        try:
            return self.vector_store.similarity_search(query, k=self.config.k_retrieval)
        except Exception as e:
            console.print(f"[red]Error during retrieval: {e!s}[/red]")
            return []
            
    @classmethod
    def list_available_collections(cls) -> list[str]:
        """List all available knowledge base collections."""
        base_dir = os.path.join(os.getcwd(), "knowledge_bases")
        if not os.path.exists(base_dir):
            return []
        return [d for d in os.listdir(base_dir) 
                if os.path.isdir(os.path.join(base_dir, d))]
                
    def get_collection_info(self) -> dict:
        """Get information about the current collection."""
        if not self.vector_store:
            return {
                "name": self.collection_name,
                "document_count": 0,
                "size": "0 KB"
            }
        
        try:
            doc_count = len(self.vector_store._collection.get()["ids"])
            size = sum(
                os.path.getsize(os.path.join(self.persist_directory, f))
                for f in os.listdir(self.persist_directory)
                if os.path.isfile(os.path.join(self.persist_directory, f))
            )
            return {
                "name": self.collection_name,
                "document_count": doc_count,
                "size": f"{size / 1024:.2f} KB"
            }
        except Exception as e:
            console.print(f"[red]Error getting collection info: {e}[/red]")
            return {
                "name": self.collection_name,
                "document_count": "Unknown",
                "size": "Unknown"
            }

    def benchmark_retrieval(self, test_queries: list[str], relevant_docs: list[list[str]], k: int = 4) -> dict:
        """Benchmark retrieval performance."""
        if not self.vector_store:
            raise ValueError("No documents ingested yet!")
            
        metrics = {
            "precision_at_k": [],
            "recall_at_k": [],
            "retrieval_time": []
        }
        
        for query, relevant in zip(test_queries, relevant_docs, strict=True):
            start_time = time.time()
            retrieved = self.vector_store.similarity_search(query, k=k)
            retrieval_time = time.time() - start_time
            
            retrieved_ids = [str(doc.metadata.get("source", "")) for doc in retrieved]
            relevant_set = set(relevant)
            retrieved_set = set(retrieved_ids)
            
            # Calculate metrics
            true_positives = len(relevant_set.intersection(retrieved_set))
            precision = true_positives / k if k > 0 else 0
            recall = true_positives / len(relevant_set) if relevant_set else 0
            
            metrics["precision_at_k"].append(precision)
            metrics["recall_at_k"].append(recall)
            metrics["retrieval_time"].append(retrieval_time)
        
        # Calculate averages
        return {
            "avg_precision_at_k": sum(metrics["precision_at_k"]) / len(test_queries),
            "avg_recall_at_k": sum(metrics["recall_at_k"]) / len(test_queries),
            "avg_retrieval_time": sum(metrics["retrieval_time"]) / len(test_queries),
            "model_name": self.config.local_embedding_model,
            "embedding_dimensions": self.config.local_dimensions
        }

    def reset_vector_store(self) -> None:
        """Reset the vector store using Chroma's native reset method and clear the in-memory reference."""
        if self.vector_store:
            try:
                self.vector_store._client.reset()
                console.print("[green]Vector store reset successfully.[/green]")
            except Exception as e:
                console.print(f"[red]Error resetting vector store: {e!s}[/red]")
            self.vector_store = None


class Agent:
    def __init__(self, name: str, role: str, config: RAGConfig):
        self.name = name
        self.role = role
        self.config = config
        self.state = {}


class SearchAgent(Agent):
    def __init__(self, config: RAGConfig):
        super().__init__("search", "retrieval", config)

    def retrieve(self, query: str, knowledge_base: KnowledgeBase) -> list[str]:
        retrieved_docs = knowledge_base.retrieve(query)
        return retrieved_docs


class ReasoningAgent(Agent):
    def __init__(self, config: RAGConfig):
        super().__init__("reason", "analysis", config)
        self.client = DeepseekQuery(config)

    def analyze(self, query: str, retrieved_docs: list[str]) -> str:
        context = "\n".join([str(doc) for doc in retrieved_docs])
        system_prompt = "You are a reasoning agent that analyzes information."
        user_prompt = (
            f"Analyze the following information in relation to the query.\n"
            f"Query: {query}\n"
            f"Context: {context}\n"
            "Provide a coherent analysis focusing on the most relevant information."
        )
        response = self.client.get_deepseek_response(system_prompt, user_prompt)
        return response


class ResponseAgent(Agent):
    def __init__(self, config: RAGConfig):
        super().__init__("response", "generation", config)
        self.client = DeepseekQuery(config)

    def generate(self, query: str, analysis: str) -> str:
        system_prompt = "You are a response agent that generates clear and comprehensive answers."
        response_prompt = (
            "Generate a comprehensive response to the query based on the analysis.\n"
            f"Query: {query}\n\n"
            f"Analysis:\n{analysis}\n\n"
            "Provide a clear and well-structured response."
        )
        response = self.client.get_deepseek_response(system_prompt, response_prompt)
        return response


class ReflectiveLayer:
    def __init__(self, config: RAGConfig):
        self.performance_metrics = []
        self.strategy_history = []
        self.config = config

    def evaluate_response(self, query: str, response: str, retrieved_docs: list[str]) -> float:
        relevance_score = self._calculate_relevance(query, retrieved_docs)
        coherence_score = self._calculate_coherence(response)
        self.performance_metrics.append({"query": query, "relevance": relevance_score, "coherence": coherence_score})
        return (relevance_score + coherence_score) / 2

    def _calculate_relevance(self, query: str, docs: list[str]) -> float:
        return np.random.uniform(0.7, 1.0)

    def _calculate_coherence(self, response: str) -> float:
        return np.random.uniform(0.7, 1.0)

    def adjust_strategy(self, current_performance: float) -> dict[str, Any]:
        if current_performance < PERFORMANCE_THRESHOLD:
            new_strategy = {"k_retrieval": self.config.k_retrieval + 2, "rewrite_query": True}
        else:
            new_strategy = {"k_retrieval": self.config.k_retrieval, "rewrite_query": False}
        self.strategy_history.append(new_strategy)
        return new_strategy


class AgentOrchestrator:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.agents = {
            "search": SearchAgent(config),
            "reason": ReasoningAgent(config),
            "response": ResponseAgent(config),
        }

    def coordinate(self, query: str, knowledge_base: KnowledgeBase, reflective_layer: ReflectiveLayer) -> str:
        retrieved_docs = self.agents["search"].retrieve(query, knowledge_base)
        if not retrieved_docs:
            return "No relevant information found."
            
        analysis = self.agents["reason"].analyze(query, retrieved_docs)
        response = self.agents["response"].generate(query, analysis)
        performance = reflective_layer.evaluate_response(query, response, retrieved_docs)
        strategy = reflective_layer.adjust_strategy(performance)
        
        # Apply the new strategy if needed
        if strategy["rewrite_query"]:
            # Attempt retrieval with adjusted parameters
            self.config.k_retrieval = strategy["k_retrieval"]
            retrieved_docs = self.agents["search"].retrieve(query, knowledge_base)
            if retrieved_docs:
                analysis = self.agents["reason"].analyze(query, retrieved_docs)
                response = self.agents["response"].generate(query, analysis)
        
        return response


class ReflectiveRAG:
    def __init__(self, config: RAGConfig, collection_name: str = "default"):
        self.config = config
        self.knowledge_base = KnowledgeBase(config, collection_name)
        self.query_processor = QueryProcessor()
        self.reflective_layer = ReflectiveLayer(config)
        self.orchestrator = AgentOrchestrator(config)

    def load_sample_dataset(self):
        dataset = load_dataset("squad", split="train[:1000]")
        contexts = [item["context"] for item in dataset]
        self.knowledge_base.ingest_documents(contexts)

    def answer_query(self, query: str) -> str:
        with console.status("[cyan]Processing query...") as status:
            status.update("[cyan]Understanding query intent...")
            query_info = self.query_processor.understand_query(query)
            console.print(
                f"[green]Detected query intent: {query_info['intent']} "
                f"(confidence: {query_info['confidence']:.2f})[/green]"
            )
            
            status.update("[cyan]Decomposing query...")
            sub_queries = self.query_processor.decompose_query(query_info)
            if len(sub_queries) > 1:
                console.print(f"[green]Query decomposed into {len(sub_queries)} sub-queries[/green]")
            
            # Process each sub-query and combine results
            responses = []
            for i, sub_query in enumerate(sub_queries, 1):
                if len(sub_queries) > 1:
                    status.update(f"[cyan]Processing sub-query {i}/{len(sub_queries)}...")
                    console.print(f"\n[blue]Sub-query {i}:[/blue] {sub_query}")
                
                response = self.orchestrator.coordinate(
                    query=sub_query,
                    knowledge_base=self.knowledge_base,
                    reflective_layer=self.reflective_layer,
                )
                responses.append(response)
            
            # If we have multiple responses, combine them
            if len(responses) > 1:
                final_response = "\n\n".join([f"Sub-query {i+1}:\n{resp}" for i, resp in enumerate(responses)])
            else:
                final_response = responses[0]
            
            console.print("[green]Query processing complete[/green]")
            return final_response
