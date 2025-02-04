import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from datasets import load_dataset
from langchain.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# Updated import for embeddings from the new package
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from transformers import pipeline


@dataclass
class RAGConfig:
    llm_api_key: str  # Your LLM API key
    llm_api_base_url: str  # Your LLM API base URL
    llm_api_version: str = "2024-01-01"  # API version if required
    llm_deployment_name: str = None  # For Azure deployment

    # Embedding Configuration
    embedding_api_key: str = None  # If using API-based embeddings
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"  # Default to local model

    # RAG Configuration
    chunk_size: int = 500
    chunk_overlap: int = 50
    k_retrieval: int = 4

    def __post_init__(self):
        if not self.llm_api_key or not self.llm_api_base_url:
            raise ValueError("LLM API key and base URL are required")


class Deepseek_query:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {config.llm_api_key}"}

    def get_deepseek_response(self, system_content, user_content):
        client = OpenAI(api_key=self.config.llm_api_key, base_url=self.config.llm_api_base_url)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            max_tokens=1024,
            temperature=0.7,
            stream=False,
        )
        if response:
            return response.choices[0].message.content
        else:
            raise Exception(f"Error {response.status_code}: {response.text}")


class QueryProcessor:
    def __init__(self):
        self.intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def understand_query(self, query: str) -> Dict[str, Any]:
        intents = self.intent_classifier(query, candidate_labels=["factual", "analytical", "procedural"])
        return {"original_query": query, "intent": intents["labels"][0], "confidence": intents["scores"][0]}

    def decompose_query(self, query_info: Dict[str, Any]) -> List[str]:
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
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
        )

    def ingest_documents(self, documents: List[str], source_type: str = "text"):
        if source_type == "text":
            docs = []
            for doc in documents:
                with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as temp_file:
                    temp_file.write(doc)
                    temp_file_path = temp_file.name
                docs.extend(TextLoader(temp_file_path, encoding="utf-8").load())
                os.remove(temp_file_path)
        elif source_type == "pdf":
            docs = [PyPDFLoader(doc).load() for doc in documents]
        split_docs = self.text_splitter.split_documents(docs)
        self.vector_store = Chroma.from_documents(documents=split_docs, embedding=self.embeddings)

    def retrieve(self, query: str) -> List[str]:
        if not self.vector_store:
            raise ValueError("No documents ingested yet!")
        return self.vector_store.similarity_search(query, k=self.config.k_retrieval)


class Agent:
    def __init__(self, name: str, role: str, config: RAGConfig):
        self.name = name
        self.role = role
        self.config = config
        self.state = {}


class SearchAgent(Agent):
    def __init__(self, config: RAGConfig):
        super().__init__("search", "retrieval", config)

    def retrieve(self, query: str, knowledge_base: KnowledgeBase) -> List[str]:
        retrieved_docs = knowledge_base.retrieve(query)
        return retrieved_docs


class ReasoningAgent(Agent):
    def __init__(self, config: RAGConfig):
        super().__init__("reason", "analysis", config)
        self.client = Deepseek_query(config)

    def analyze(self, query: str, retrieved_docs: List[str]) -> str:
        context = "\n".join([str(doc) for doc in retrieved_docs])
        system_prompt = "You are a reasoning agent that analyzes information."
        user_prompt = f"Analyze the following information in relation to the query. \n Query: {query} \n Context: {context} \n Provide a coherent analysis focusing on the most relevant information."
        response = self.client.get_deepseek_response(system_prompt, user_prompt)
        return response


class ResponseAgent(Agent):
    def __init__(self, config: RAGConfig):
        super().__init__("response", "generation", config)
        self.client = Deepseek_query(config)

    def generate(self, query: str, analysis: str) -> str:
        system_prompt = "You are a response agent that generates clear and comprehensive answers."
        response_prompt = f"""Generate a comprehensive response to the query based on the analysis.\nQuery: {query}\n\nAnalysis:\n{analysis}\n\nProvide a clear and well-structured response."""
        response = self.client.get_deepseek_response(system_prompt, response_prompt)
        return response


class ReflectiveLayer:
    def __init__(self, config: RAGConfig):
        self.performance_metrics = []
        self.strategy_history = []
        self.config = config

    def evaluate_response(self, query: str, response: str, retrieved_docs: List[str]) -> float:
        relevance_score = self._calculate_relevance(query, retrieved_docs)
        coherence_score = self._calculate_coherence(response)
        self.performance_metrics.append({"query": query, "relevance": relevance_score, "coherence": coherence_score})
        return (relevance_score + coherence_score) / 2

    def _calculate_relevance(self, query: str, docs: List[str]) -> float:
        return np.random.uniform(0.7, 1.0)

    def _calculate_coherence(self, response: str) -> float:
        return np.random.uniform(0.7, 1.0)

    def adjust_strategy(self, current_performance: float) -> Dict[str, Any]:
        if current_performance < 0.8:
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
        new_strategy = reflective_layer.adjust_strategy(performance)
        return response


class ReflectiveRAG:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.query_processor = QueryProcessor()
        self.knowledge_base = KnowledgeBase(config)
        self.reflective_layer = ReflectiveLayer(config)
        self.orchestrator = AgentOrchestrator(config)

    def load_sample_dataset(self):
        dataset = load_dataset("squad", split="train[:1000]")
        contexts = [item["context"] for item in dataset]
        self.knowledge_base.ingest_documents(contexts)

    def answer_query(self, query: str) -> str:
        query_info = self.query_processor.understand_query(query)
        sub_queries = self.query_processor.decompose_query(query_info)
        response = self.orchestrator.coordinate(
            query=query_info["original_query"],
            knowledge_base=self.knowledge_base,
            reflective_layer=self.reflective_layer,
        )
        return response
