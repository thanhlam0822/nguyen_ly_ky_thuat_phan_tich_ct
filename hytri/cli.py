"""HyTri Command Line Interface.

Provides commands for:
- triage: Run failure triage on a build
- evaluate: Evaluate triage results against ground truth
- benchmark: Build and run benchmarks
"""

import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from hytri.benchmark.builder import MultiFaultBuildBuilder
from hytri.benchmark.loader import BugsInPyLoader
from hytri.evaluation import (
    evaluate_clustering,
    evaluate_explanation_groundedness,
    evaluate_prioritization,
)
from hytri.models import EvaluationResult
from hytri.pipeline import HyTriPipeline

app = typer.Typer(
    name="hytri",
    help="HyTri - Hybrid Failure Triage System",
    add_completion=False,
)
console = Console()


@app.command()
def triage(
    input_file: Path = typer.Argument(
        ...,
        help="Path to failure instances JSONL file",
        exists=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for triage results (JSON)",
    ),
    alpha: float = typer.Option(
        0.5,
        "--alpha",
        "-a",
        help="Similarity fusion weight (0=semantic only, 1=lexical only)",
    ),
    min_cluster_size: int = typer.Option(
        2,
        "--min-cluster-size",
        "-m",
        help="Minimum cluster size for HDBSCAN",
    ),
    no_semantic: bool = typer.Option(
        False,
        "--no-semantic",
        help="Disable LLM-based semantic similarity",
    ),
    no_explanations: bool = typer.Option(
        False,
        "--no-explanations",
        help="Disable explanation generation",
    ),
    clustering_mode: str = typer.Option(
        "ensemble",
        "--clustering-mode",
        help="Clustering mode: ensemble (default), change_aware, category_guided, fused, two_stage",
    ),
) -> None:
    """Run failure triage on a build."""
    console.print(Panel.fit("HyTri - Hybrid Failure Triage", style="bold blue"))

    # Load failures
    loader = BugsInPyLoader()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Loading failures...", total=None)
        failures = loader.load_from_jsonl(input_file)

    console.print(f"Loaded [bold]{len(failures)}[/bold] failures")

    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not no_semantic:
        console.print(
            "[yellow]Warning: OPENAI_API_KEY not set, using lexical-only mode[/yellow]"
        )
        no_semantic = True

    # Run triage
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Running triage pipeline...", total=None)

        pipeline = HyTriPipeline(
            openai_api_key=api_key,
            clustering_mode=clustering_mode,
            alpha=alpha,
            min_cluster_size=min_cluster_size,
            use_semantic=not no_semantic,
            generate_explanations=not no_explanations,
        )
        result = pipeline.triage(failures)

    # Display results
    console.print()
    console.print(f"[bold green]Triage complete![/bold green]")
    console.print(f"  Clusters: {len(result.clusters)}")
    console.print(f"  Noise failures: {result.num_noise_failures}")
    console.print()

    # Show cluster summary
    table = Table(title="Cluster Summary (Priority Order)")
    table.add_column("Rank", style="dim")
    table.add_column("Cluster ID")
    table.add_column("Failures", justify="right")
    table.add_column("Exception Types")
    table.add_column("Title")
    table.add_column("Score", justify="right")

    cluster_map = {c.cluster_id: c for c in result.clusters}
    for i, cluster_id in enumerate(result.priority_queue[:10], 1):
        cluster = cluster_map.get(cluster_id)
        if not cluster:
            continue

        title = cluster.title or "[no title]"
        if len(title) > 40:
            title = title[:37] + "..."

        table.add_row(
            str(i),
            cluster.cluster_id[:8],
            str(len(cluster.failure_ids)),
            ", ".join(cluster.exception_types[:2]),
            title,
            f"{cluster.severity_score:.2f}",
        )

    console.print(table)

    # Save output
    if output:
        with open(output, "w") as f:
            json.dump(result.model_dump(), f, indent=2, default=str)
        console.print(f"\nResults saved to [bold]{output}[/bold]")


@app.command()
def evaluate(
    results_file: Path = typer.Argument(
        ...,
        help="Path to triage results JSON file",
        exists=True,
    ),
    ground_truth: Path = typer.Option(
        ...,
        "--ground-truth",
        "-g",
        help="Path to ground truth JSONL file",
        exists=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for evaluation results (JSON)",
    ),
) -> None:
    """Evaluate triage results against ground truth."""
    console.print(Panel.fit("HyTri Evaluation", style="bold blue"))

    # Load triage results
    with open(results_file) as f:
        results_data = json.load(f)

    # Load ground truth failures
    loader = BugsInPyLoader()
    failures = loader.load_from_jsonl(ground_truth)

    # Reconstruct clusters from results
    from hytri.models import Cluster, ClusterExplanation

    clusters = []
    for cluster_data in results_data.get("clusters", []):
        explanation_data = cluster_data.get("explanation")
        explanation = (
            ClusterExplanation(**explanation_data) if explanation_data else None
        )
        cluster = Cluster(
            cluster_id=cluster_data["cluster_id"],
            failure_ids=cluster_data["failure_ids"],
            representative_ids=cluster_data.get("representative_ids", []),
            explanation=explanation,
            severity_score=cluster_data.get("severity_score", 0.0),
            exception_types=cluster_data.get("exception_types", []),
            top_frames=cluster_data.get("top_frames", []),
        )
        clusters.append(cluster)

    priority_queue = results_data.get("priority_queue", [])
    noise_ids = results_data.get("noise_failure_ids", [])

    # Compute metrics
    console.print("Computing metrics...")

    clustering_metrics = evaluate_clustering(clusters, failures, noise_ids)
    prioritization_metrics = evaluate_prioritization(clusters, priority_queue, failures)
    explanation_metrics = evaluate_explanation_groundedness(clusters, failures)

    # Display results
    console.print()

    # Clustering metrics
    table = Table(title="Clustering Metrics")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Adjusted Rand Index (ARI)", f"{clustering_metrics.ari:.3f}")
    table.add_row("Normalized Mutual Info (NMI)", f"{clustering_metrics.nmi:.3f}")
    table.add_row("V-measure", f"{clustering_metrics.v_measure:.3f}")
    table.add_row("Purity", f"{clustering_metrics.purity:.3f}")
    table.add_row(
        "Cluster count error",
        f"{clustering_metrics.cluster_count_error} ({clustering_metrics.num_predicted_clusters} vs {clustering_metrics.num_true_clusters})",
    )

    console.print(table)
    console.print()

    # Prioritization metrics
    table = Table(title="Prioritization Metrics")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("nDCG@1", f"{prioritization_metrics.ndcg_at_1:.3f}")
    table.add_row("nDCG@3", f"{prioritization_metrics.ndcg_at_3:.3f}")
    table.add_row("nDCG@5", f"{prioritization_metrics.ndcg_at_5:.3f}")
    table.add_row("Top-1 Coverage", f"{prioritization_metrics.top_1_coverage:.1%}")
    table.add_row("Top-3 Coverage", f"{prioritization_metrics.top_3_coverage:.1%}")
    table.add_row("Top-5 Coverage", f"{prioritization_metrics.top_5_coverage:.1%}")

    console.print(table)
    console.print()

    # Explanation metrics
    table = Table(title="Explanation Metrics")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Avg Groundedness", f"{explanation_metrics.avg_groundedness:.3f}")
    table.add_row("Verified Ratio", f"{explanation_metrics.verified_ratio:.1%}")
    table.add_row("Coverage", f"{explanation_metrics.coverage:.1%}")

    console.print(table)

    # Save output
    if output:
        eval_result = EvaluationResult(
            ari=clustering_metrics.ari,
            nmi=clustering_metrics.nmi,
            v_measure=clustering_metrics.v_measure,
            purity=clustering_metrics.purity,
            cluster_count_error=clustering_metrics.cluster_count_error,
            ndcg_at_1=prioritization_metrics.ndcg_at_1,
            ndcg_at_3=prioritization_metrics.ndcg_at_3,
            ndcg_at_5=prioritization_metrics.ndcg_at_5,
            top_1_coverage=prioritization_metrics.top_1_coverage,
            top_3_coverage=prioritization_metrics.top_3_coverage,
            top_5_coverage=prioritization_metrics.top_5_coverage,
            avg_groundedness=explanation_metrics.avg_groundedness,
            explanation_coverage=explanation_metrics.coverage,
        )

        with open(output, "w") as f:
            json.dump(eval_result.model_dump(), f, indent=2)

        console.print(f"\nEvaluation results saved to [bold]{output}[/bold]")


# Benchmark subcommand group
benchmark_app = typer.Typer(help="Benchmark commands")
app.add_typer(benchmark_app, name="benchmark")


@benchmark_app.command("build")
def benchmark_build(
    project: str = typer.Option(
        "pandas",
        "--project",
        "-p",
        help="BugsInPy project to use",
    ),
    num_builds: int = typer.Option(
        50,
        "--num-builds",
        "-n",
        help="Number of synthetic builds to generate",
    ),
    output_dir: Path = typer.Option(
        Path("benchmark"),
        "--output",
        "-o",
        help="Output directory for benchmark",
    ),
    bugsinpy_path: Optional[Path] = typer.Option(
        None,
        "--bugsinpy-path",
        help="Path to BugsInPy repository",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Random seed for reproducibility",
    ),
) -> None:
    """Build a benchmark dataset from BugsInPy."""
    console.print(Panel.fit("HyTri Benchmark Builder", style="bold blue"))

    loader = BugsInPyLoader(bugsinpy_path=bugsinpy_path)

    # Check if project exists
    projects = loader.list_projects()
    if project not in projects:
        console.print(f"[red]Project '{project}' not found.[/red]")
        console.print(f"Available projects: {', '.join(projects[:10])}...")
        raise typer.Exit(1)

    console.print(f"Building benchmark for project: [bold]{project}[/bold]")
    console.print(f"Number of builds: {num_builds}")

    # List bugs
    bugs = loader.list_bugs(project)
    if not bugs:
        console.print(
            "[yellow]No bugs found. Make sure BugsInPy path is correct.[/yellow]"
        )
        console.print(
            "You can download BugsInPy from: https://github.com/soarsmu/BugsInPy"
        )
        raise typer.Exit(1)

    console.print(f"Found {len(bugs)} bugs")

    # For now, create a placeholder benchmark
    # In real usage, you would run tests and collect failures
    console.print(
        "[yellow]Note: Full benchmark generation requires running tests.[/yellow]"
    )
    console.print("Creating placeholder benchmark structure...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create sample data structure
    sample_data = {
        "project": project,
        "num_bugs": len(bugs),
        "num_builds": num_builds,
        "bugs": bugs[:10],  # First 10 bugs as sample
    }

    with open(output_dir / "benchmark_info.json", "w") as f:
        json.dump(sample_data, f, indent=2)

    console.print(f"\nBenchmark info saved to [bold]{output_dir}[/bold]")
    console.print(
        "\nTo generate full benchmark with failures, use the Python API:\n"
        "  from hytri.benchmark import BugsInPyLoader, MultiFaultBuildBuilder\n"
        "  loader = BugsInPyLoader(bugsinpy_path='path/to/bugsinpy')\n"
        "  # ... load failures and build benchmark"
    )


@benchmark_app.command("run")
def benchmark_run(
    input_dir: Path = typer.Argument(
        ...,
        help="Path to benchmark directory",
        exists=True,
    ),
    baselines: str = typer.Option(
        "all",
        "--baselines",
        "-b",
        help="Baselines to run (all, lexical, semantic, hytri)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for results",
    ),
) -> None:
    """Run baselines on a benchmark."""
    console.print(Panel.fit("HyTri Benchmark Runner", style="bold blue"))

    # Load benchmark
    failures_path = input_dir / "failure_instances.jsonl"
    if not failures_path.exists():
        console.print(f"[red]No failure_instances.jsonl found in {input_dir}[/red]")
        raise typer.Exit(1)

    loader = BugsInPyLoader()
    failures = loader.load_from_jsonl(failures_path)

    console.print(f"Loaded {len(failures)} failures from benchmark")

    # Determine baselines to run
    baseline_list = []
    if baselines == "all":
        baseline_list = ["lexical", "semantic", "hytri"]
    else:
        baseline_list = [b.strip() for b in baselines.split(",")]

    results = {}

    for baseline_name in baseline_list:
        console.print(f"\nRunning baseline: [bold]{baseline_name}[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(f"Running {baseline_name}...", total=None)

            if baseline_name == "lexical":
                pipeline = HyTriPipeline(
                    alpha=1.0,  # Lexical only
                    use_semantic=False,
                    generate_explanations=False,
                )
            elif baseline_name == "semantic":
                pipeline = HyTriPipeline(
                    alpha=0.0,  # Semantic only
                    use_semantic=True,
                    generate_explanations=False,
                )
            else:  # hytri
                pipeline = HyTriPipeline(
                    alpha=0.5,
                    use_semantic=True,
                    generate_explanations=True,
                )

            result = pipeline.triage(failures)

        # Evaluate
        clustering_metrics = evaluate_clustering(
            result.clusters, failures, result.noise_failure_ids
        )

        results[baseline_name] = {
            "num_clusters": len(result.clusters),
            "ari": clustering_metrics.ari,
            "nmi": clustering_metrics.nmi,
            "v_measure": clustering_metrics.v_measure,
        }

    # Display comparison
    console.print()
    table = Table(title="Baseline Comparison")
    table.add_column("Baseline")
    table.add_column("Clusters", justify="right")
    table.add_column("ARI", justify="right")
    table.add_column("NMI", justify="right")
    table.add_column("V-measure", justify="right")

    for name, metrics in results.items():
        table.add_row(
            name,
            str(metrics["num_clusters"]),
            f"{metrics['ari']:.3f}",
            f"{metrics['nmi']:.3f}",
            f"{metrics['v_measure']:.3f}",
        )

    console.print(table)

    # Save results
    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"\nResults saved to [bold]{output}[/bold]")


@app.command()
def version() -> None:
    """Show version information."""
    from hytri import __version__

    console.print(f"HyTri version: [bold]{__version__}[/bold]")


if __name__ == "__main__":
    app()


