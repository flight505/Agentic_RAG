import os
import time
from pathlib import Path
from typing import ClassVar

import questionary
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.status import Status
from rich.panel import Panel

# Import our RAG system with correct package path
from agentic_rag.rag_system import KnowledgeBase, RAGConfig, ReflectiveRAG, DEVICE

app = typer.Typer(
    name="agentic-rag",
    help="Agentic RAG System - Intelligent document processing and querying",
    add_completion=False
)
console = Console()

# Global state to track loaded PDFs and RAG system instance
_rag_instance: ReflectiveRAG | None = None
_loaded_pdfs: set[str] = set()


class RAGSingleton:
    """Singleton class to manage RAG system instance."""

    _instances: ClassVar[dict[str, ReflectiveRAG]] = {}
    _current_collection: ClassVar[str] = "default"
    _initialized: ClassVar[set[str]] = set()  # Track which collections have been initialized

    @classmethod
    def get_instance(cls, collection_name: str | None = None) -> ReflectiveRAG:
        """Get or create RAG system instance for a specific collection."""
        if collection_name is None:
            collection_name = cls._current_collection

        if collection_name not in cls._instances:
            with Status(f"[bold blue]Initializing collection '{collection_name}'...", spinner="dots") as status:
                # Try Kluster AI first, fallback to Deepseek
                kluster_key = os.getenv("KLUSTER_API_KEY")
                deepseek_key = os.getenv("DEEPSEEK_API_KEY")
                
                if kluster_key:
                    config = RAGConfig(
                        llm_api_key=kluster_key,
                        llm_api_base_url="https://api.kluster.ai/v1",
                    )
                elif deepseek_key:
                    config = RAGConfig(
                        llm_api_key=deepseek_key,
                        llm_api_base_url="https://api.deepseek.com/v1",
                    )
                else:
                    console.print("[red]Error: No API key found for either Kluster AI or Deepseek[/red]")
                    raise typer.Exit(1)
                
                cls._instances[collection_name] = ReflectiveRAG(config, collection_name)
                cls._initialized.add(collection_name)
                console.print(Panel(f"Collection '{collection_name}' initialized successfully", style="green"))
        
        return cls._instances[collection_name]

    @classmethod
    def set_current_collection(cls, collection_name: str) -> None:
        """Set the current active collection."""
        cls._current_collection = collection_name
        # Initialize the collection if needed
        cls.get_instance(collection_name)


def get_rag_system(collection_name: str | None = None) -> ReflectiveRAG:
    """Initialize the RAG system with configuration."""
    return RAGSingleton.get_instance(collection_name)


def load_pdf_with_progress(pdf_path: str, progress: Progress) -> None:
    """Load a PDF file with progress tracking."""
    pdf_path = str(Path(pdf_path).resolve())

    # Check if PDF is already loaded
    if pdf_path in _loaded_pdfs:
        console.print(Panel(f"PDF already loaded: {Path(pdf_path).name}", style="yellow"))
        return

    task = progress.add_task(f"[cyan]Processing {Path(pdf_path).name}", total=100)

    try:
        # Start PDF conversion and ingestion
        progress.update(task, advance=10, description=f"[cyan]Preparing {Path(pdf_path).name}[/cyan]")
        
        rag = get_rag_system()
        
        # The actual conversion progress will be shown through our Rich display
        rag.knowledge_base.ingest_pdf(pdf_path)
        
        # Update progress after conversion is complete
        progress.update(task, completed=100)
        
        console.print(Panel(
            f"Successfully loaded: {Path(pdf_path).name}\n"
            f"Collection: {RAGSingleton._current_collection}",
            style="green",
            title="Success"
        ))
        _loaded_pdfs.add(pdf_path)
        
    except Exception as e:
        progress.update(task, description=f"[red]Error processing {Path(pdf_path).name}")
        console.print(Panel(f"Error loading PDF: {str(e)}", style="red", title="Error"))
        raise typer.Exit(1) from e


def show_loaded_pdfs():
    """Display currently loaded PDFs."""
    if not _loaded_pdfs:
        console.print(Panel("No PDFs currently loaded", style="yellow"))
    else:
        pdfs = "\n".join([f"• {Path(pdf).name}" for pdf in sorted(_loaded_pdfs)])
        console.print(Panel(
            f"Collection: {RAGSingleton._current_collection}\n\n{pdfs}",
            title="Loaded PDFs",
            style="blue"
        ))


def handle_single_pdf() -> bool:
    """Handle loading a single PDF file."""
    pdf_path = questionary.text(
        "Enter the path to your PDF file:", validate=lambda text: Path(text).exists() or "File not found"
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
        "Enter the directory containing PDFs:", validate=lambda text: Path(text).is_dir() or "Directory not found"
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

        selected_files = questionary.checkbox("Select PDFs to load:", choices=[str(pdf) for pdf in new_pdfs]).ask()

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
    actions = [
        "Load Single PDF",
        "Load Multiple PDFs",
        "View Loaded PDFs",
        questionary.Separator(),
        "← Back",
    ]

    while True:
        console.print(f"\n[bold cyan]Current Collection: {RAGSingleton._current_collection}[/bold cyan]")
        action = questionary.select("PDF Loading Menu:", choices=actions).ask()

        if action == "← Back":
            return True
        elif action == "Load Single PDF":
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
            except Exception:
                if not questionary.select(
                    "Error loading PDF. What would you like to do?",
                    choices=["Try Again", "← Back"]
                ).ask() == "Try Again":
                    continue

        elif action == "Load Multiple PDFs":
            pdf_dir = questionary.text(
                "Enter the directory containing PDFs:",
                validate=lambda text: Path(text).is_dir() or "Directory not found"
            ).ask()
            try:
                pdf_files = list(Path(pdf_dir).glob("*.pdf"))
                if not pdf_files:
                    console.print("[yellow]No PDF files found in directory[/yellow]")
                    continue

                # Filter out already loaded PDFs
                new_pdfs = [pdf for pdf in pdf_files if str(pdf.resolve()) not in _loaded_pdfs]
                if not new_pdfs:
                    console.print("[yellow]All PDFs in this directory are already loaded[/yellow]")
                    continue

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
            except Exception:
                if not questionary.select(
                    "Error loading PDFs. What would you like to do?",
                    choices=["Try Again", "← Back"]
                ).ask() == "Try Again":
                    continue

        elif action == "View Loaded PDFs":
            show_loaded_pdfs()


def handle_querying() -> bool:
    """Handle querying operation. Returns True if operation should continue."""
    while True:
        action = questionary.select(
            "Query Menu:",
            choices=[
                "Ask Question",
                "View Previous Questions",
                questionary.Separator(),
                "← Back"
            ]
        ).ask()

        if action == "← Back":
            return True
        elif action == "Ask Question":
            question = questionary.text(
                "Enter your question:",
                validate=lambda text: len(text.strip()) > 0 or "Question cannot be empty"
            ).ask()
            try:
                with Status("[bold blue]Processing...", spinner="dots") as status:
                    status.update("[bold blue]Searching knowledge base...")
                    rag = get_rag_system()
                    status.update("[bold blue]Generating response...")
                    response = rag.answer_query(question)
                    console.print(f"\n[green]Answer:[/green] {response}")
                
                if questionary.select(
                    "What would you like to do next?",
                    choices=["Ask Another Question", "← Back"]
                ).ask() == "← Back":
                    continue
                    
            except Exception as e:
                console.print(f"[red]Error: {e!s}[/red]")
                if questionary.select(
                    "What would you like to do?",
                    choices=["Try Again", "← Back"]
                ).ask() == "← Back":
                    continue
                    
        elif action == "View Previous Questions":
            with Status("[bold blue]Loading history...", spinner="dots"):
                rag = get_rag_system()
                metrics = rag.reflective_layer.performance_metrics
                if not metrics:
                    console.print("[yellow]No previous questions found[/yellow]")
                else:
                    console.print("\n[blue]Previous Questions:[/blue]")
                    for i, metric in enumerate(metrics, 1):
                        console.print(
                            f"{i}. [cyan]{metric['query']}[/cyan]\n"
                            f"   Relevance: {metric['relevance']:.2f} | "
                            f"Coherence: {metric['coherence']:.2f}"
                        )


def show_collections():
    """Display available collections and their information."""
    collections = KnowledgeBase.list_available_collections()
    if not collections:
        console.print("[yellow]No collections available[/yellow]")
        return

    console.print("\n[blue]Available Collections:[/blue]")
    for collection in collections:
        rag = RAGSingleton.get_instance(collection)
        info = rag.knowledge_base.get_collection_info()
        is_active = collection == RAGSingleton._current_collection
        console.print(
            f"{'[cyan]' if is_active else ''}{info['name']}"
            f"{' [active]' if is_active else ''}"
            f" ({info['document_count']} docs, {info['size']})"
            f"{'[/cyan]' if is_active else ''}"
        )


def handle_collection_management() -> bool:
    """Handle collection management operations."""
    while True:
        console.print(f"\n[bold cyan]Current Collection: {RAGSingleton._current_collection}[/bold cyan]")
        action = questionary.select(
            "Collection Management:",
            choices=[
                "Create New Collection",
                "Switch Collection",
                "View Collections",
                questionary.Separator(),
                "← Back",
            ],
        ).ask()

        if action == "← Back":
            return True
        elif action == "Create New Collection":
            collection_name = questionary.text(
                "Enter name for new collection:",
                validate=lambda text: bool(text.strip()) or "Collection name cannot be empty",
            ).ask()

            if collection_name:
                RAGSingleton.set_current_collection(collection_name)
                console.print(f"[green]Created and switched to: {collection_name}[/green]")

        elif action == "Switch Collection":
            collections = KnowledgeBase.list_available_collections()
            if not collections:
                console.print("[yellow]No collections available. Create one first.[/yellow]")
                continue

            collection = questionary.select(
                "Select collection:",
                choices=[*collections, questionary.Separator(), "← Back"]
            ).ask()
            
            if collection == "← Back":
                continue
                
            RAGSingleton.set_current_collection(collection)
            console.print(f"[green]Switched to: {collection}[/green]")

        elif action == "View Collections":
            show_collections()


def show_help():
    """Display help information about the system."""
    help_text = """
[bold blue]Agentic RAG System Help[/bold blue]

[cyan]Overview:[/cyan]
This system combines Retrieval-Augmented Generation (RAG) with agentic capabilities for intelligent document processing and querying.

[cyan]Key Features:[/cyan]
• Multiple Collections: Organize documents in separate knowledge bases
• PDF Processing: Advanced PDF conversion with layout preservation
• Smart Querying: Intelligent query decomposition and analysis
• Performance Metrics: Track and optimize system performance

[cyan]Tips for Optimal Use:[/cyan]
1. Collections
   • Create separate collections for different topics/projects
   • Use meaningful collection names for easy navigation

2. PDF Loading
   • Ensure PDFs are text-searchable for best results
   • Large PDFs (>50MB) may take longer to process
   • The system preserves document layout and handles equations

3. Querying
   • Be specific in your questions for better results
   • Complex queries are automatically broken down
   • Use follow-up questions to refine results

4. Performance
   • Monitor relevance and coherence scores
   • System automatically adjusts retrieval strategy
   • MPS acceleration is enabled on Apple Silicon

[cyan]Device Information:[/cyan]
• Current Device: [green]{DEVICE}[/green]
• Optimized for: Apple Silicon (when available)
"""
    console.print(Panel(help_text, title="System Help", border_style="blue"))


def handle_metrics_view() -> bool:
    """Handle viewing performance metrics. Returns True to continue, False to exit."""
    with Status("[bold blue]Loading performance metrics...", spinner="dots"):
        rag = get_rag_system()
        metrics = rag.reflective_layer.performance_metrics
        if not metrics:
            console.print("[yellow]No performance metrics available yet[/yellow]")
        else:
            console.print("\n[blue]Performance Metrics:[/blue]")
            for i, metric in enumerate(metrics, 1):
                console.print(
                    f"[cyan]Query {i}:[/cyan] {metric['query']}\n"
                    f"  Relevance: {metric['relevance']:.2f} | "
                    f"Coherence: {metric['coherence']:.2f}"
                )

    return questionary.select(
        "What would you like to do?",
        choices=["View More Details", "← Back"]
    ).ask() == "View More Details"


@app.command()
def main_menu():
    """Run the main menu interface"""
    console.print(Panel.fit(
        "[bold green]Welcome to Agentic RAG System[/bold green]\n"
        "[cyan]Your intelligent document assistant powered by RAG technology[/cyan]\n"
        "Type '← Help' at any time to access system information",
        title="👋 Welcome",
        border_style="green"
    ))
    
    menu_items = [
        "Manage Collections",
        "Load PDFs",
        "Query System",
        "View Performance Metrics",
        questionary.Separator(),
        "← Help",
        questionary.Separator(),
        "Exit"
    ]
    
    while True:
        console.print(f"\n[bold cyan]Current Collection: {RAGSingleton._current_collection}[/bold cyan]")
        action = questionary.select("Main Menu:", choices=menu_items).ask()

        if action == "Exit":
            console.print(Panel.fit(
                "[green]Thank you for using Agentic RAG System![/green]\n"
                "Your session has been saved.",
                border_style="green"
            ))
            break
        elif action == "← Help":
            show_help()
        elif action == "Manage Collections":
            if not handle_collection_management():
                break
        elif action == "Load PDFs":
            if not handle_pdf_loading():
                break
        elif action == "Query System":
            if not handle_querying():
                break
        elif action == "View Performance Metrics":
            if not handle_metrics_view():
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


@app.command()
def help():
    """Show system help information"""
    show_help()


@app.command()
def version():
    """Show system version information"""
    console.print(Panel.fit(
        "[bold green]Agentic RAG System[/bold green]\n"
        "Version: 1.0.0\n"
        f"Running on: {DEVICE}\n"
        "Made with ❤️ for document intelligence",
        title="Version Info",
        border_style="blue"
    ))


if __name__ == "__main__":
    app()
