import os
import time
from pathlib import Path

import questionary
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.status import Status

# Import our RAG system with correct package path
from agentic_rag.rag_system import RAGConfig, ReflectiveRAG

app = typer.Typer()
console = Console()

# Global state to track loaded PDFs and RAG system instance
_rag_instance: ReflectiveRAG | None = None
_loaded_pdfs: set[str] = set()

class RAGSingleton:
    """Singleton class to manage RAG system instance."""
    _instance: ReflectiveRAG | None = None
    
    @classmethod
    def get_instance(cls) -> ReflectiveRAG:
        """Get or create RAG system instance."""
        if cls._instance is None:
            with Status("[bold blue]Initializing RAG system...", spinner="dots") as status:
                config = RAGConfig(
                    llm_api_key=os.getenv("DEEPSEEK_API_KEY"),
                    llm_api_base_url="https://api.deepseek.com/v1",
                )
                status.update("[bold blue]Loading models and embeddings...")
                cls._instance = ReflectiveRAG(config)
        return cls._instance

def get_rag_system() -> ReflectiveRAG:
    """Initialize the RAG system with configuration."""
    return RAGSingleton.get_instance()

def load_pdf_with_progress(pdf_path: str, progress: Progress) -> None:
    """Load a PDF file with progress tracking."""
    pdf_path = str(Path(pdf_path).resolve())
    
    # Check if PDF is already loaded
    if pdf_path in _loaded_pdfs:
        console.print(f"[yellow]PDF already loaded: {pdf_path}[/yellow]")
        return
    
    task = progress.add_task(f"[cyan]Processing {Path(pdf_path).name}", total=100)
    
    # Simulate progress steps for PDF processing
    progress.update(task, advance=20, description=f"[cyan]Loading {Path(pdf_path).name}")
    rag = get_rag_system()
    
    progress.update(task, advance=30, description=f"[cyan]Extracting text from {Path(pdf_path).name}")
    try:
        rag.knowledge_base.ingest_pdf(pdf_path)
        progress.update(task, advance=50, description=f"[cyan]Finalizing {Path(pdf_path).name}")
        time.sleep(0.5)  # Give user time to see completion
        progress.update(task, completed=100)
        console.print(f"[green]Successfully loaded {pdf_path}[/green]")
        _loaded_pdfs.add(pdf_path)
    except Exception as e:
        progress.update(task, description=f"[red]Error processing {Path(pdf_path).name}")
        console.print(f"[red]Error loading PDF: {e!s}[/red]")
        raise typer.Exit(1) from e

def show_loaded_pdfs():
    """Display currently loaded PDFs."""
    if not _loaded_pdfs:
        console.print("[yellow]No PDFs currently loaded[/yellow]")
    else:
        console.print("[blue]Currently loaded PDFs:[/blue]")
        for pdf in _loaded_pdfs:
            console.print(f"  - {Path(pdf).name}")

def handle_single_pdf() -> bool:
    """Handle loading a single PDF file."""
    pdf_path = questionary.text(
        "Enter the path to your PDF file:",
        validate=lambda text: Path(text).exists() or "File not found"
    ).ask()
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            load_pdf_with_progress(pdf_path, progress)
        return True
    except Exception:
        return questionary.confirm("Would you like to try again?").ask()

def handle_multiple_pdfs() -> bool:
    """Handle loading multiple PDF files."""
    pdf_dir = questionary.text(
        "Enter the directory containing PDFs:",
        validate=lambda text: Path(text).is_dir() or "Directory not found"
    ).ask()
    try:
        pdf_files = list(Path(pdf_dir).glob("*.pdf"))
        if not pdf_files:
            console.print("[yellow]No PDF files found in directory[/yellow]")
            return True
        
        # Filter out already loaded PDFs
        new_pdfs = [pdf for pdf in pdf_files if str(pdf.resolve()) not in _loaded_pdfs]
        if not new_pdfs:
            console.print("[yellow]All PDFs in this directory are already loaded[/yellow]")
            return True
        
        selected_files = questionary.checkbox(
            "Select PDFs to load:",
            choices=[str(pdf) for pdf in new_pdfs]
        ).ask()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            for pdf_path in selected_files:
                load_pdf_with_progress(pdf_path, progress)
        return True
    except Exception:
        console.print("[red]Error loading PDFs[/red]")
        return questionary.confirm("Would you like to try again?").ask()

def handle_pdf_loading() -> bool:
    """Handle PDF loading operation. Returns True if operation should continue."""
    actions = {
        "Load Single PDF": handle_single_pdf,
        "Load Multiple PDFs": handle_multiple_pdfs,
        "View Loaded PDFs": lambda: (show_loaded_pdfs(), True)[1],
        "Back to Main Menu": lambda: True,
        "Exit": lambda: False
    }
    
    while True:
        action = questionary.select(
            "PDF Loading Menu:",
            choices=list(actions.keys())
        ).ask()
        
        if action == "Exit":
            console.print("[yellow]Exiting system...[/yellow]")
            raise typer.Exit()
            
        result = actions[action]()
        if action != "View Loaded PDFs" and not result:
            return False

def handle_querying() -> bool:
    """Handle querying operation. Returns True if operation should continue."""
    while True:
        action = questionary.select(
            "Query Menu:",
            choices=[
                "Ask Question",
                "View Previous Questions",
                "Back to Main Menu",
                "Exit"
            ]
        ).ask()
        
        if action == "Exit":
            console.print("[yellow]Exiting system...[/yellow]")
            raise typer.Exit()
        elif action == "Back to Main Menu":
            return True
        elif action == "Ask Question":
            question = questionary.text(
                "Enter your question:",
                validate=lambda text: len(text.strip()) > 0 or "Question cannot be empty"
            ).ask()
            try:
                with Status("[bold blue]Processing your question...", spinner="dots") as status:
                    status.update("[bold blue]Searching knowledge base...")
                    rag = get_rag_system()
                    status.update("[bold blue]Generating response...")
                    response = rag.answer_query(question)
                    console.print(f"[green]Response:[/green] {response}")
                if not questionary.confirm("Would you like to ask another question?").ask():
                    return True
            except Exception as e:
                console.print(f"[red]Error processing query: {e!s}[/red]")
                if not questionary.confirm("Would you like to try again?").ask():
                    return True
        elif action == "View Previous Questions":
            with Status("[bold blue]Loading question history...", spinner="dots"):
                rag = get_rag_system()
                metrics = rag.reflective_layer.performance_metrics
                if not metrics:
                    console.print("[yellow]No previous questions found[/yellow]")
                else:
                    console.print("[blue]Previous Questions:[/blue]")
                    for i, metric in enumerate(metrics, 1):
                        console.print(f"{i}. Query: {metric['query']}")
                        console.print(f"   Relevance: {metric['relevance']:.2f}")
                        console.print(f"   Coherence: {metric['coherence']:.2f}\n")
            
            if not questionary.confirm("Return to Query Menu?").ask():
                return True

@app.command()
def main_menu():
    """Run the main menu interface"""
    console.print("[bold green]Welcome to Agentic RAG System[/bold green]")
    
    while True:
        action = questionary.select(
            "Main Menu:",
            choices=[
                "Load PDFs",
                "Query System",
                "View Performance Metrics",
                "Exit"
            ]
        ).ask()
        
        if action == "Exit":
            console.print("[green]Thank you for using Agentic RAG System![/green]")
            break
        elif action == "Load PDFs":
            if not handle_pdf_loading():
                break
        elif action == "Query System":
            if not handle_querying():
                break
        elif action == "View Performance Metrics":
            with Status("[bold blue]Loading performance metrics...", spinner="dots"):
                rag = get_rag_system()
                metrics = rag.reflective_layer.performance_metrics
                if not metrics:
                    console.print("[yellow]No performance metrics available yet[/yellow]")
                else:
                    console.print("[blue]Performance Metrics:[/blue]")
                    for i, metric in enumerate(metrics, 1):
                        console.print(f"Query {i}:")
                        console.print(f"  Relevance: {metric['relevance']:.2f}")
                        console.print(f"  Coherence: {metric['coherence']:.2f}")
            
            if not questionary.confirm("Return to Main Menu?").ask():
                console.print("[green]Thank you for using Agentic RAG System![/green]")
                break

@app.command()
def query(question: str):
    """Query the RAG system"""
    with Status("[bold blue]Processing query...", spinner="dots") as status:
        rag = get_rag_system()
        status.update("[bold blue]Generating response...")
        response = rag.answer_query(question)
        console.print(f"[green]Response:[/green] {response}")

@app.command()
def load_pdf(pdf_path: str):
    """Load a PDF file"""
    path = Path(pdf_path)
    if not path.exists():
        console.print(f"[red]Error: File {pdf_path} not found[/red]")
        raise typer.Exit(1)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        load_pdf_with_progress(pdf_path, progress)

if __name__ == "__main__":
    app() 