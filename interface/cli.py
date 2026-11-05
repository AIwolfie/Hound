"""HOUND Forensic Command Line Interface.

Distinctive terminal identity for visual geolocation and OSINT investigations.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional
import click
from rich.box import DOUBLE, ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from core.configuration import get_config, load_config
from core.image_loader import load_and_assess_image
from core.logger import get_console, setup_logger
from investigators.metadata.exif_analyzer import ExifAnalyzer
from storage.database import DatabaseManager
from storage.evidence_store import EvidenceStore
from storage.investigation_store import InvestigationStore

# Ensure standard UTF-8 stream output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

console = get_console()


def print_banner() -> None:
    """Render HOUND's signature forensic terminal header."""
    banner_text = Text(
        "╔══════════════════════════════════════════╗\n"
        "║                 H O U N D                ║\n"
        "║      VISUAL GEOLOCATION INTELLIGENCE     ║\n"
        "║          EVERY IMAGE LEAVES A TRAIL      ║\n"
        "╚══════════════════════════════════════════╝",
        style="bold cyan",
    )
    console.print(banner_text)
    console.print()


@click.group()
@click.option("--config", "-c", type=click.Path(exists=True), help="Custom config YAML path.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose debug logging.")
def cli(config: Optional[str], verbose: bool) -> None:
    """HOUND — Autonomous Visual Geolocation & OSINT Agent."""
    cfg = load_config(config)
    log_level = "DEBUG" if verbose else cfg.hound.log_level
    setup_logger(log_level=log_level)


@cli.command("intake")
@click.argument("image_path", type=click.Path(exists=True, path_type=Path))
@click.option("--no-preserve", is_flag=True, help="Do not copy image to storage directory.")
def cmd_intake(image_path: Path, no_preserve: bool) -> None:
    """Execute Phase 1 Intake & Forensic Analysis on an image."""
    print_banner()
    cfg = get_config()
    db_manager = DatabaseManager(cfg.paths.database_path)
    inv_store = InvestigationStore(db_manager, cfg.paths.storage_dir)
    evid_store = EvidenceStore(db_manager)

    console.print(f"[bold green][+][/bold green] Target acquired: [bold]{image_path.name}[/bold]")
    console.print("[dim][*] Initializing investigation session...[/dim]")

    t0 = time.perf_counter()
    try:
        image_record = load_and_assess_image(
            image_path=image_path,
            storage_dir=cfg.paths.storage_dir,
            preserve_original=not no_preserve,
        )
    except Exception as e:
        console.print(f"[bold red][!] Ingestion failure:[/bold red] {e}")
        sys.exit(1)

    inv = inv_store.create_investigation(image_record)
    console.print(f"[bold green][+][/bold green] Investigation initialized: [cyan]{inv.id}[/cyan]")
    console.print(f"[bold green][+][/bold green] SHA-256: [yellow]{image_record.sha256}[/yellow]")
    console.print(
        f"[bold green][+][/bold green] Resolution: {image_record.width}x{image_record.height} "
        f"({image_record.format}, {image_record.file_size_bytes:,} bytes)"
    )

    # Run Metadata Investigator
    console.print("[dim][*] Executing metadata investigator...[/dim]")
    analyzer = ExifAnalyzer()
    evidence_items = analyzer.run(inv.id, image_record)

    for item in evidence_items:
        evid_store.add_evidence(item)

    dt = time.perf_counter() - t0
    console.print(f"[bold green][+][/bold green] Metadata analysis complete ({dt:.2f}s)")

    # Print Summary Table
    table = Table(title="Target Forensic Profile", box=ROUNDED)
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Observation", style="white")

    table.add_row("Filename", image_record.filename)
    table.add_row("SHA-256", image_record.sha256[:16] + "...")
    table.add_row("Resolution", f"{image_record.width} x {image_record.height} ({image_record.aspect_ratio:.2f}:1)")
    table.add_row("Integrity Status", f"[{'green' if image_record.integrity.status == 'complete' else 'yellow'}]{image_record.integrity.status.upper()}[/]")
    
    if image_record.exif.gps:
        gps_str = image_record.exif.gps.to_formatted_str()
        table.add_row("Embedded GPS", f"[bold green]{gps_str}[/bold green]")
        if image_record.exif.gps.altitude_m is not None:
            table.add_row("Altitude", f"{image_record.exif.gps.altitude_m:.1f} m")
        if image_record.exif.gps.heading_deg is not None:
            table.add_row("Camera Heading", f"{image_record.exif.gps.heading_deg:.1f}°")
    else:
        table.add_row("Embedded GPS", "[red]None (Stripped or unavailable)[/red]")

    if image_record.exif.make or image_record.exif.model:
        table.add_row("Camera Hardware", f"{image_record.exif.make or ''} {image_record.exif.model or ''}".strip())
    if image_record.exif.focal_length_35mm:
        table.add_row("Optics", f"{image_record.exif.focal_length_35mm}mm (35mm equiv)")
    if image_record.exif.datetime_original:
        table.add_row("Capture Timestamp", image_record.exif.datetime_original.strftime("%Y-%m-%d %H:%M:%S"))

    console.print(table)

    # Print Findings Box
    findings_text = "\n".join(f"• {f}" for f in image_record.integrity.findings)
    console.print(Panel(findings_text, title="Forensic Observations", border_style="blue"))


@cli.command("inspect")
@click.argument("image_path", type=click.Path(exists=True, path_type=Path))
def cmd_inspect(image_path: Path) -> None:
    """Deep forensic inspection of image metadata and raw EXIF tags."""
    print_banner()
    try:
        record = load_and_assess_image(image_path, preserve_original=False)
    except Exception as e:
        console.print(f"[bold red][!] Error inspecting image:[/bold red] {e}")
        sys.exit(1)

    console.print(f"[bold cyan]Forensic Inspection:[/bold cyan] {image_path.name}")
    console.print(f"Format: {record.format} | Mode: {record.mode} | Dimensions: {record.width}x{record.height}")
    console.print(f"SHA-256: {record.sha256}")

    if not record.exif.raw_tags:
        console.print("[yellow]No raw EXIF tags discovered.[/yellow]")
        return

    table = Table(title=f"Raw EXIF Tags ({len(record.exif.raw_tags)} tags)", box=ROUNDED)
    table.add_column("Tag Name", style="cyan")
    table.add_column("Value", style="white")

    for k, v in sorted(record.exif.raw_tags.items()):
        val_str = str(v)
        if len(val_str) > 80:
            val_str = val_str[:77] + "..."
        table.add_row(k, val_str)

    console.print(table)


@cli.command("doctor")
@click.option("--network", is_flag=True, help="Test external service reachability.")
def cmd_doctor(network: bool) -> None:
    """Run HOUND environment, dependency, and network diagnostics."""
    print_banner()
    cfg = get_config()
    console.print("[bold]Running HOUND System Diagnostics...[/bold]\n")

    # 1. Local environment checks
    console.print(f"Python Runtime: [green]{sys.version.split()[0]}[/green]")
    console.print(f"Workspace Directory: {cfg.paths.workspace_dir}")
    console.print(f"Database Path: {cfg.paths.database_path}")

    # Check Geo-Sleuth path
    gs_path = cfg.paths.geo_sleuth_path
    if gs_path.is_dir():
        console.print(f"[green][PASS][/green] Geo-Sleuth repository: Found at {gs_path}")
    else:
        console.print(f"[yellow][WARN][/yellow] Geo-Sleuth repository: Not found at {gs_path}")

    # Check database writable
    try:
        db = DatabaseManager(cfg.paths.database_path)
        with db.get_connection() as conn:
            conn.execute("SELECT 1")
        console.print("[green][PASS][/green] SQLite database: Initialized and writable")
    except Exception as e:
        console.print(f"[red][FAIL][/red] SQLite database: {e}")

    # Check dependencies
    for pkg in ["PIL", "pydantic", "rich", "yaml", "torch"]:
        try:
            __import__(pkg)
            console.print(f"[green][PASS][/green] Dependency '{pkg}': Available")
        except ImportError:
            console.print(f"[red][FAIL][/red] Dependency '{pkg}': Missing")

    # Optional network probes
    if network:
        import urllib.request

        console.print("\n[bold]Testing Network Services...[/bold]")
        endpoints = [
            ("Elevation Tiles", "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/0/0/0.png"),
            ("Google Satellite", "https://mt1.google.com/vt/lyrs=s&x=0&y=0&z=0"),
            ("Hugging Face", "https://huggingface.co"),
            ("Yandex Images", "https://yandex.com/images/"),
            ("OSM Nominatim", "https://nominatim.openstreetmap.org/search?q=Paris&format=json&limit=1"),
        ]

        for name, url in endpoints:
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": cfg.network.user_agent},
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    status = resp.status
                    console.print(f"[green][PASS][/green] {name}: HTTP {status}")
            except Exception as e:
                console.print(f"[yellow][WARN][/yellow] {name}: {e}")


@cli.command("list")
def cmd_list() -> None:
    """List recorded investigations."""
    print_banner()
    cfg = get_config()
    db_manager = DatabaseManager(cfg.paths.database_path)
    inv_store = InvestigationStore(db_manager, cfg.paths.storage_dir)

    investigations = inv_store.list_investigations(limit=25)
    if not investigations:
        console.print("[dim]No investigations recorded in database.[/dim]")
        return

    table = Table(title="Investigation History", box=ROUNDED)
    table.add_column("Investigation ID", style="cyan")
    table.add_column("Target", style="white")
    table.add_column("Status", style="green")
    table.add_column("Started", style="dim")
    table.add_column("SHA-256", style="yellow")

    for inv in investigations:
        table.add_row(
            inv["id"],
            inv["filename"],
            inv["status"].upper(),
            inv["started_at"][:19],
            inv["sha256"][:12] + "...",
        )

    console.print(table)


def main() -> None:
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
