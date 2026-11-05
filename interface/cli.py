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
from core.image_preprocessor import ImagePreprocessor
from core.logger import get_console, setup_logger
from investigators.infrastructure.infrastructure_analyzer import InfrastructureAnalyzer
from investigators.metadata.exif_analyzer import ExifAnalyzer
from investigators.ocr.ocr_engine import OCREngine
from investigators.shadows.solar_analyzer import SolarAnalyzer
from reports.evidence_visualizer import EvidenceVisualizer
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
    """Execute Phase 1 Intake & Forensic Metadata Analysis on an image."""
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
    table.add_row(
        "Integrity Status",
        f"[{'green' if image_record.integrity.status == 'complete' else 'yellow'}]{image_record.integrity.status.upper()}[/]",
    )

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
    findings_text = "\n".join(f"• {f}" for f in image_record.integrity.findings)
    console.print(Panel(findings_text, title="Forensic Observations", border_style="blue"))


@cli.command("preprocess")
@click.argument("image_path", type=click.Path(exists=True, path_type=Path))
@click.option("--output-dir", "-o", type=click.Path(path_type=Path), help="Output directory for variants.")
def cmd_preprocess(image_path: Path, output_dir: Optional[Path]) -> None:
    """Generate normalized resolution variants, contrast/edge enhancements, and coordinate tiles."""
    print_banner()
    cfg = get_config()
    out_dir = output_dir or (cfg.paths.storage_dir / "preprocessed" / image_path.stem)

    console.print(f"[bold green][+][/bold green] Target acquired: [bold]{image_path.name}[/bold]")
    console.print(f"[dim][*] Preprocessing image variants into {out_dir}...[/dim]")

    preprocessor = ImagePreprocessor()
    t0 = time.perf_counter()
    manifest = preprocessor.process(image_path, out_dir, generate_tiles=True, generate_edges=True)
    dt = time.perf_counter() - t0

    console.print(f"[bold green][+][/bold green] Preprocessing complete ({dt:.2f}s)")
    console.print(f"Original: {manifest.original_dimensions} -> Normalized: {manifest.normalized_dimensions}")

    table = Table(title="Generated Analysis Variants", box=ROUNDED)
    table.add_column("Variant Name", style="cyan")
    table.add_column("Dimensions", style="yellow")
    table.add_column("Description", style="white")

    for v in manifest.analysis_variants:
        table.add_row(v.name, f"{v.dimensions[0]}x{v.dimensions[1]}", v.description)

    console.print(table)
    console.print(f"[dim]Generated {len(manifest.crops)} spatial crops/tiles in {out_dir / 'crops'}[/dim]")


@cli.command("ocr")
@click.argument("image_path", type=click.Path(exists=True, path_type=Path))
@click.option("--min-conf", type=float, default=0.30, help="Minimum recognition confidence threshold.")
def cmd_ocr(image_path: Path, min_conf: float) -> None:
    """Execute multi-pass OCR investigation, coordinate mapping, and clue classification."""
    print_banner()
    cfg = get_config()
    console.print(f"[bold green][+][/bold green] Target acquired: [bold]{image_path.name}[/bold]")
    console.print("[dim][*] Initializing multi-pass OCR engine...[/dim]")

    t0 = time.perf_counter()
    image_record = load_and_assess_image(image_path, storage_dir=cfg.paths.storage_dir, preserve_original=False)
    ocr_engine = OCREngine(min_confidence=min_conf)

    workspace_temp = cfg.paths.storage_dir / "temp_ocr" / image_record.sha256[:10]
    workspace_temp.mkdir(parents=True, exist_ok=True)

    items = ocr_engine.run(
        investigation_id="cli_ocr_session",
        image_record=image_record,
        context={"workspace_dir": workspace_temp},
    )
    dt = time.perf_counter() - t0

    console.print(f"[bold green][+][/bold green] OCR investigation complete ({dt:.2f}s) — [bold]{len(items)}[/bold] text clues detected")

    if not items:
        console.print("[dim]No text detected meeting confidence threshold.[/dim]")
        return

    table = Table(title="Recognized Textual Evidence", box=ROUNDED)
    table.add_column("Text Observation", style="bold white")
    table.add_column("Confidence", style="yellow")

    for item in items:
        table.add_row(item.text, f"{item.confidence_weight:.2f}")

    console.print(table)


@cli.command("analyze")
@click.argument("image_path", type=click.Path(exists=True, path_type=Path))
def cmd_analyze(image_path: Path) -> None:
    """Run full Phase 2 Visual Intelligence pipeline (Intake + Preprocessing + OCR + Infrastructure + Shadows)."""
    print_banner()
    console.print("[bold cyan]HOUND VISUAL INVESTIGATION[/bold cyan]\n")
    cfg = get_config()

    t_start = time.perf_counter()
    console.print(f"[bold green][+][/bold green] Target acquired: [bold]{image_path.name}[/bold]")

    db_manager = DatabaseManager(cfg.paths.database_path)
    inv_store = InvestigationStore(db_manager, cfg.paths.storage_dir)
    evid_store = EvidenceStore(db_manager)

    # 1. Image Intake & EXIF
    image_record = load_and_assess_image(
        image_path=image_path,
        storage_dir=cfg.paths.storage_dir,
        preserve_original=True,
    )
    inv = inv_store.create_investigation(image_record)
    workspace = inv.workspace_dir
    overlays_dir = workspace / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    # 2. Metadata Investigator
    meta_analyzer = ExifAnalyzer()
    meta_items = meta_analyzer.run(inv.id, image_record)
    for m in meta_items:
        evid_store.add_evidence(m)

    # 3. Image Preprocessing
    preprocessor = ImagePreprocessor()
    preprocessed_dir = workspace / "preprocessed"
    manifest = preprocessor.process(image_record.original_path, preprocessed_dir, generate_tiles=True, generate_edges=True)
    console.print("[bold green][+][/bold green] Image preprocessing complete")

    # 4. OCR Investigation
    ocr_engine = OCREngine()
    ocr_items = ocr_engine.run(
        inv.id,
        image_record,
        context={"workspace_dir": workspace, "db_manager": db_manager},
    )
    for o in ocr_items:
        evid_store.add_evidence(o)
    console.print(f"[bold green][+][/bold green] OCR investigation complete ([cyan]{len(ocr_items)}[/cyan] text observations)")

    # 5. Infrastructure Analysis
    infra_analyzer = InfrastructureAnalyzer()
    infra_items = infra_analyzer.run(
        inv.id,
        image_record,
        context={"db_manager": db_manager},
    )
    for inf in infra_items:
        evid_store.add_evidence(inf)
    console.print(f"[bold green][+][/bold green] Infrastructure analysis complete ([cyan]{len(infra_items)}[/cyan] feature classes)")

    # 6. Solar and Shadow Analysis
    solar_analyzer = SolarAnalyzer()
    solar_items = solar_analyzer.run(
        inv.id,
        image_record,
        context={"db_manager": db_manager},
    )
    for s in solar_items:
        evid_store.add_evidence(s)
    console.print(f"[bold green][+][/bold green] Shadow analysis attempted ([cyan]{len(solar_items)}[/cyan] solar observations)")

    # 7. Generate Visual Overlays
    visualizer = EvidenceVisualizer()

    # Preprocessing comparison sheet
    comp_sheet = overlays_dir / "comparison_sheet.jpg"
    visualizer.generate_comparison_panel(manifest.analysis_variants, comp_sheet)

    # OCR Overlay if text was found
    with db_manager.get_connection() as conn:
        cur = conn.execute("SELECT * FROM ocr_observations WHERE investigation_id = ?", (inv.id,))
        obs_rows = cur.fetchall()
        if obs_rows:
            from investigators.ocr.ocr_engine import OCRObservation

            obs_list = [
                OCRObservation(
                    investigation_id=r["investigation_id"],
                    text=r["text"],
                    confidence=r["confidence"],
                    box_original=(r["box_x0"], r["box_y0"], r["box_x1"], r["box_y1"]),
                    category=r["category"],
                    language_script=r["language_script"],
                    pass_name=r["pass_name"],
                )
                for r in obs_rows
            ]
            ocr_overlay_path = overlays_dir / "ocr_overlay.jpg"
            visualizer.draw_ocr_overlay(image_record.original_path, obs_list, ocr_overlay_path)

        # Infrastructure Overlay
        cur_infra = conn.execute("SELECT * FROM infrastructure_detections WHERE investigation_id = ?", (inv.id,))
        infra_rows = cur_infra.fetchall()
        if infra_rows:
            from investigators.infrastructure.base_detector import DetectedFeature

            feats = [
                DetectedFeature(
                    class_name=r["class_name"],
                    confidence=r["confidence"],
                    box=(r["box_x0"], r["box_y0"], r["box_x1"], r["box_y1"]),
                    detector_backend=r["detector_backend"],
                )
                for r in infra_rows
            ]
            infra_overlay_path = overlays_dir / "infrastructure_overlay.jpg"
            visualizer.draw_infrastructure_overlay(image_record.original_path, feats, infra_overlay_path)

    total_time = time.perf_counter() - t_start
    inv_store.update_status(inv.id, "visual_analyzed", summary=f"Phase 2 visual intelligence complete in {total_time:.2f}s")

    # Display Visual Intelligence Summary
    console.print("\n[bold cyan]VISUAL INTELLIGENCE SUMMARY[/bold cyan]")
    summary_table = Table(box=ROUNDED)
    summary_table.add_column("Investigation Domain", style="cyan")
    summary_table.add_column("Observations Extracted", style="bold green")

    summary_table.add_row("Text & OCR Observations", str(len(ocr_items)))
    summary_table.add_row("Infrastructure Clues", str(len(infra_items)))
    summary_table.add_row("Solar & Shadow Observations", str(len(solar_items)))
    summary_table.add_row("Metadata Observations", str(len(meta_items)))
    summary_table.add_row("Geographic Hypotheses", "[dim yellow]Pending Phase 3 (Geographic Engine)[/dim yellow]")

    console.print(summary_table)
    console.print(f"[bold green][+][/bold green] Investigation artifacts saved to: [cyan]{workspace}[/cyan]")
    console.print(f"[dim]Total pipeline duration: {total_time:.2f}s[/dim]")


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

    # Check reference GIS suite path
    ref_path = cfg.paths.geo_sleuth_path
    if ref_path.is_dir():
        console.print(f"[green][PASS][/green] Reference GIS Toolkit: Verified at {ref_path}")
    else:
        console.print(f"[yellow][WARN][/yellow] Reference GIS Toolkit: Not found at {ref_path}")

    # Check database writable
    try:
        db = DatabaseManager(cfg.paths.database_path)
        with db.get_connection() as conn:
            conn.execute("SELECT 1")
        console.print("[green][PASS][/green] SQLite database: Initialized and writable")
    except Exception as e:
        console.print(f"[red][FAIL][/red] SQLite database: {e}")

    # Check dependencies
    for pkg in ["PIL", "pydantic", "rich", "yaml", "torch", "cv2", "rapidocr_onnxruntime"]:
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
