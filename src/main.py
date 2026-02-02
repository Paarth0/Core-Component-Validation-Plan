#!/usr/bin/env python3
"""
PKA Validation Core - CLI Entry Point

Usage:
    python src/main.py ingest --input ./data/sample.pdf --verbose
    python src/main.py search --query "search terms" --top-k 5
"""

import asyncio
import click
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from ingestion_coordinator import IngestionCoordinator

console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="pka-ingest")
def cli():
    """
    PKA Validation Core - Document Ingestion Pipeline
    
    Validates the iPhone 17 Pro PKA v2.0 architecture on laptop hardware.
    """
    pass


@cli.command()
@click.option(
    '--input', '-i', 'input_path',
    required=True,
    type=click.Path(exists=True),
    help='Path to file (PDF, DOCX, XLSX, XML)'
)
@click.option(
    '--output', '-o', 'output_path',
    default='./output',
    type=click.Path(),
    help='Output directory for results'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output'
)
def ingest(input_path: str, output_path: str, verbose: bool):
    """
    Ingest a document and create vector embeddings.
    
    Supports: PDF, DOCX, XLSX, XML (Apple Health)
    
    Example:
        python src/main.py ingest -i ./data/sample.pdf -v
    """
    console.print(Panel.fit(
        "[bold blue]PKA Validation Core[/bold blue]\n"
        "[dim]Laptop-based validation framework[/dim]",
        border_style="blue"
    ))
    
    console.print(f"\n📄 Processing: [cyan]{input_path}[/cyan]\n")
    
    # Run async ingestion
    result = asyncio.run(run_ingestion(input_path, output_path, verbose))
    
    # Display results
    display_results(result)
    
    # Save JSON output
    Path(output_path).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_path) / "ingestion_result.json"
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    console.print(f"\n[green]✓ Results saved to:[/green] {output_file}")
    
    # Return exit code based on errors
    if result.get('errors'):
        console.print(f"[red]⚠ Completed with errors[/red]")
        sys.exit(1)
    else:
        console.print(f"[green]✓ Ingestion complete![/green]")


async def run_ingestion(input_path: str, output_path: str, verbose: bool) -> dict:
    """Execute the ingestion pipeline."""
    coordinator = IngestionCoordinator(
        output_dir=output_path,
        verbose=verbose
    )
    return await coordinator.process(input_path)


def display_results(result: dict):
    """Display results in a formatted table."""
    table = Table(title="📊 Ingestion Results", show_header=True)
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", style="green")
    
    table.add_row("File", str(result.get('file', 'N/A')))
    table.add_row("Total Chunks", str(result.get('total_chunks', 0)))
    table.add_row("Vectors Stored", str(result.get('vectors_stored', 0)))
    table.add_row("Processing Time", f"{result.get('processing_time_sec', 0):.2f} seconds")
    table.add_row("Pages/Second", f"{result.get('pages_per_sec', 0):.2f}")
    table.add_row("Peak Memory", f"{result.get('memory_mb', 0):.1f} MB")
    
    if result.get('embedding_latency_ms'):
        table.add_row("Avg Embedding Latency", f"{result.get('embedding_latency_ms', 0):.1f} ms")
    
    if result.get('errors'):
        table.add_row("Errors", str(result['errors']), style="red")
    
    console.print(table)


@cli.command()
@click.option(
    '--query', '-q',
    required=True,
    help='Search query text'
)
@click.option(
    '--top-k', '-k',
    default=5,
    type=int,
    help='Number of results to return'
)
@click.option(
    '--db', '-d',
    default='./output/vectors.db',
    type=click.Path(exists=True),
    help='Path to vector database'
)
def search(query: str, top_k: int, db: str):
    """
    Search the vector database for similar chunks.
    
    Example:
        python src/main.py search -q "machine learning" -k 5
    """
    console.print(f"\n🔍 Searching for: [cyan]{query}[/cyan]\n")
    
    results = asyncio.run(run_search(query, top_k, db))
    
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return
    
    for i, result in enumerate(results, 1):
        console.print(Panel(
            f"[dim]{result['text'][:200]}...[/dim]" if len(result['text']) > 200 else result['text'],
            title=f"Result {i} (Score: {result['score']:.3f})",
            subtitle=f"ID: {result['id']}",
            border_style="green" if result['score'] > 0.7 else "yellow"
        ))


async def run_search(query: str, top_k: int, db_path: str) -> list:
    """Execute vector search."""
    from embedding_worker import EmbeddingWorker
    from indexing_worker import IndexingWorker
    
    # Generate query embedding
    embedder = EmbeddingWorker()
    query_embedding = await embedder.embed(query)
    
    # Search database
    indexer = IndexingWorker(db_path=db_path)
    await indexer.initialize()
    
    results = await indexer.search(query_embedding, top_k=top_k)
    
    return results


@cli.command()
@click.option(
    '--db', '-d',
    default='./output/vectors.db',
    type=click.Path(exists=True),
    help='Path to vector database'
)
def stats(db: str):
    """
    Show statistics about the vector database.
    
    Example:
        python src/main.py stats
    """
    stats = asyncio.run(get_stats(db))
    
    table = Table(title="📈 Database Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Chunks", str(stats.get('chunk_count', 0)))
    table.add_row("Total Vectors", str(stats.get('vector_count', 0)))
    table.add_row("Database Path", stats.get('db_path', 'N/A'))
    
    console.print(table)


async def get_stats(db_path: str) -> dict:
    """Get database statistics."""
    from indexing_worker import IndexingWorker
    
    indexer = IndexingWorker(db_path=db_path)
    await indexer.initialize()
    
    return await indexer.get_stats()


if __name__ == '__main__':
    cli()