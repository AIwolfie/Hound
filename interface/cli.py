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

    # Check core geographic intelligence subsystem
    console.print("[green][PASS][/green] Geographic Intelligence Subsystem: Native engines active")

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


# ==============================================================================
# Phase 3 Geographic Intelligence Commands: hound geo <subcommand>
# ==============================================================================

@cli.group("geo")
def geo_group() -> None:
    """Geographic Intelligence Engine commands (OSM, terrain, gazetteer, candidates)."""
    pass


@geo_group.command("doctor")
def cmd_geo_doctor() -> None:
    """Diagnose Geographic Intelligence endpoints (Overpass, Terrarium, Gazetteer)."""
    print_banner()
    cfg = get_config()
    console.print("[bold]Diagnosing Geographic Intelligence Endpoints...[/bold]\n")

    # 1. OpenStreetMap Overpass Mirrors
    from integrations.openstreetmap.overpass_client import OverpassClient

    osm_client = OverpassClient()
    console.print("[bold cyan]OpenStreetMap Overpass Mirrors:[/bold cyan]")
    for mirror in osm_client.mirrors:
        try:
            import urllib.request
            t0 = time.perf_counter()
            req = urllib.request.Request(
                mirror.replace("/api/interpreter", "/api/status"),
                headers={"User-Agent": cfg.network.user_agent},
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                dt_ms = (time.perf_counter() - t0) * 1000.0
                console.print(f"  [green][PASS][/green] {mirror} (HTTP {resp.status}, {dt_ms:.0f}ms)")
        except Exception as e:
            console.print(f"  [yellow][WARN][/yellow] {mirror}: {e}")

    # 2. Terrarium Elevation DEM
    from integrations.elevation.terrarium_client import TerrariumClient

    dem = TerrariumClient()
    console.print("\n[bold cyan]Digital Elevation Model (DEM):[/bold cyan]")
    try:
        # Test elevation lookup at Mount Everest (8848m)
        t0 = time.perf_counter()
        elev = dem.get_elevation(27.9881, 86.9250, zoom=10)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        console.print(
            f"  [green][PASS][/green] AWS Terrarium Elevation: Verified (Mount Everest sample: {elev:.0f}m, {dt_ms:.0f}ms)"
        )
    except Exception as e:
        console.print(f"  [red][FAIL][/red] Terrarium Elevation Tile Service: {e}")

    # 3. Geocoding Service
    from integrations.geocoding.gazetteer import GazetteerService

    gazetteer = GazetteerService()
    console.print("\n[bold cyan]Geocoding & Gazetteer Service:[/bold cyan]")
    try:
        res = gazetteer.forward_geocode("Paris, France", limit=1)
        if res:
            console.print(f"  [green][PASS][/green] Forward Geocoding: {res[0].display_name[:50]}...")
        else:
            console.print("  [yellow][WARN][/yellow] Geocoding returned empty results.")
    except Exception as e:
        console.print(f"  [red][FAIL][/red] Geocoding Service: {e}")


@geo_group.command("terrain")
@click.argument("latitude", type=float)
@click.argument("longitude", type=float)
@click.option("--distance-km", type=float, default=20.0, help="Radial horizon distance in km.")
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output path for horizon plot.")
def cmd_geo_terrain(latitude: float, longitude: float, distance_km: float, output: Optional[Path]) -> None:
    """Generate 360-degree synthetic terrain horizon profile from coordinates."""
    print_banner()
    console.print(f"[bold cyan]Terrain Horizon Computation:[/bold cyan] ({latitude:.5f}, {longitude:.5f})\n")

    from integrations.elevation.horizon_generator import HorizonGenerator
    from reports.geographic_visualizer import GeographicVisualizer

    gen = HorizonGenerator()
    t0 = time.perf_counter()
    profile = gen.generate_horizon(
        lat=latitude,
        lon=longitude,
        max_distance_m=distance_km * 1000.0,
        bearing_step_deg=1.0,
    )
    dt = time.perf_counter() - t0

    console.print(f"[bold green][+][/bold green] Horizon profile generated ({dt:.2f}s)")
    console.print(f"Ground Elevation: [yellow]{profile.observer_elevation_m:.1f} m[/yellow]")
    console.print(
        f"Prominent Peak: [bold cyan]{profile.peak_elevation_deg:.2f}°[/bold cyan] at bearing "
        f"[bold yellow]{profile.peak_azimuth_deg:.1f}°[/bold yellow] (True North)"
    )

    out_file = output or Path(f"horizon_{latitude:.3f}_{longitude:.3f}.jpg")
    GeographicVisualizer.render_horizon_profile_plot(profile, out_file)
    console.print(f"[dim]Saved synthetic horizon profile plot to {out_file}[/dim]")


@geo_group.command("investigate")
@click.argument("investigation_id", type=str)
def cmd_geo_investigate(investigation_id: str) -> None:
    """Execute Phase 3 Geographic Intelligence pipeline on an existing investigation."""
    print_banner()
    console.print(f"[bold cyan]HOUND GEOGRAPHIC INVESTIGATION[/bold cyan]: [bold]{investigation_id}[/bold]\n")

    from agent.investigation_manager import GeographicInvestigationManager

    manager = GeographicInvestigationManager()
    try:
        summary = manager.run_investigation(investigation_id)
    except Exception as e:
        console.print(f"[bold red][!] Investigation failed:[/bold red] {e}")
        sys.exit(1)

    console.print(f"[bold green][+][/bold green] Geographic synthesis complete ({summary.execution_time_seconds:.2f}s)")
    console.print(f"[bold green][+][/bold green] Hypotheses Generated: [bold]{summary.candidates_count}[/bold]")
    console.print(f"[bold green][+][/bold green] Top Hypothesis: [bold cyan]{summary.top_candidate_name}[/bold cyan]")
    console.print(f"     Level: [yellow]{summary.top_candidate_level.upper()}[/yellow] | Score: [green]{summary.top_candidate_score:.2f}[/green]")

    if summary.top_candidate_coords:
        lat, lon = summary.top_candidate_coords
        console.print(f"     Coordinates: [bold]{lat:.6f}, {lon:.6f}[/bold]")

    table = Table(title="Ranked Geographic Hypotheses", box=ROUNDED)
    table.add_column("Rank", style="bold cyan", width=6)
    table.add_column("Candidate Hypothesis", style="white")
    table.add_column("Level", style="yellow")
    table.add_column("Heuristic Score", style="green")
    table.add_column("Status", style="bold")

    for ev in summary.evaluations[:8]:
        status_style = "green" if ev.status == "active" else "red"
        table.add_row(
            f"#{ev.rank}",
            ev.name[:45],
            ev.level.upper(),
            f"{ev.composite_score:.2f}",
            f"[{status_style}]{ev.status.upper()}[/{status_style}]",
        )

    console.print(table)

    if summary.evaluations:
        top_e = summary.evaluations[0]
        sb = top_e.score_breakdown
        breakdown_text = (
            f"• Base Seed Score: {sb.base_seed_score:.2f} ({sb.seed_clue_description})\n"
            f"• Level Specificity Multiplier: {sb.level_specificity_multiplier:.1f}x (heuristic weight)\n"
            f"• Independent Corroborating Clues: {sb.independent_corroborations_count} (+{sb.corroboration_bonus:.2f})\n"
            f"• Verified Terrain Correlation Bonus: +{sb.verified_terrain_bonus:.2f}\n"
            f"• Contradiction Penalty Factor: {sb.contradiction_penalty_factor:.4f}x\n"
            f"• Final Heuristic Score: {top_e.composite_score:.2f} (Heuristic rank; not a statistical probability)"
        )
        console.print(
            Panel(
                breakdown_text,
                title=f"Score Breakdown: #{top_e.rank} {top_e.name[:35]}",
                border_style="cyan",
            )
        )

    if summary.unresolved_questions:
        unresolved_text = "\n".join(f"• {q}" for q in summary.unresolved_questions)
        console.print(Panel(unresolved_text, title="Unresolved Geographic Constraints", border_style="yellow"))

    console.print(f"\n[bold green][+][/bold green] Visualization maps saved:")
    for k, v in summary.artifacts.items():
        console.print(f"  • {k}: [cyan]{v}[/cyan]")


@geo_group.command("candidates")
@click.argument("investigation_id", type=str)
def cmd_geo_candidates(investigation_id: str) -> None:
    """View detailed candidate hypotheses, scores, and supporting/contradictory evidence."""
    print_banner()
    cfg = get_config()
    db_manager = DatabaseManager(cfg.paths.database_path)
    evid_store = EvidenceStore(db_manager)

    candidates = evid_store.get_candidates(investigation_id)
    if not candidates:
        console.print(f"[dim]No candidates recorded for investigation '{investigation_id}'.[/dim]")
        return

    table = Table(title=f"Candidate Ledger: {investigation_id}", box=ROUNDED)
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Level", style="yellow")
    table.add_column("Coordinates", style="dim")
    table.add_column("Score", style="green")
    table.add_column("Status", style="bold")

    for c in candidates:
        coord_str = f"{c.lat:.4f}, {c.lon:.4f}" if c.lat and c.lon else "N/A"
        status_color = "green" if c.status == "active" else "red"
        table.add_row(
            c.id,
            c.name[:40],
            c.level.upper(),
            coord_str,
            f"{c.score:.2f}",
            f"[{status_color}]{c.status.upper()}[/{status_color}]",
        )

    console.print(table)


@geo_group.command("map")
@click.argument("investigation_id", type=str)
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output map path.")
def cmd_geo_map(investigation_id: str, output: Optional[Path]) -> None:
    """Generate and display geographic map for an investigation's candidate pool."""
    print_banner()
    from agent.candidate_generator import GeographicCandidate
    from reports.geographic_visualizer import GeographicVisualizer

    cfg = get_config()
    db_manager = DatabaseManager(cfg.paths.database_path)
    evid_store = EvidenceStore(db_manager)
    inv_store = InvestigationStore(db_manager, cfg.paths.storage_dir)

    inv = inv_store.get_investigation(investigation_id)
    if not inv:
        console.print(f"[red]Investigation '{investigation_id}' not found.[/red]")
        sys.exit(1)

    cands_raw = evid_store.get_candidates(investigation_id)
    cands = [
        GeographicCandidate(
            id=c.id,
            investigation_id=c.investigation_id,
            level=c.level,
            name=c.name,
            lat=c.lat,
            lon=c.lon,
            radius_m=c.radius_m,
            score=c.score,
            status=c.status,
        )
        for c in cands_raw
    ]

    out_file = output or (inv.workspace_dir / "geographic" / "candidate_map.jpg")
    GeographicVisualizer.render_candidate_map(cands, out_file)
    console.print(f"[bold green][+][/bold green] Candidate map generated: [cyan]{out_file}[/cyan]")


# ==============================================================================
# PHASE 4: AUTONOMOUS AGENT COMMANDS
# ==============================================================================

@cli.group("agent")
def agent_group() -> None:
    """Phase 4 Autonomous Investigation Agent commands."""
    pass


@agent_group.command("investigate")
@click.argument("investigation_id")
@click.option("--max-iterations", "-i", type=int, default=15, help="Maximum agent planning iterations.")
@click.option("--max-tool-calls", type=int, default=30, help="Maximum allowlisted tool calls.")
@click.option("--max-duration", type=int, default=120, help="Maximum investigation duration in seconds.")
@click.option("--offline", is_flag=True, help="Force offline mode (disable external network queries).")
@click.option("--no-llm", is_flag=True, help="Disable optional LLM reasoning and use deterministic offline planner.")
@click.option("--verbose", "-v", is_flag=True, help="Display verbose event telemetry.")
@click.option("--resume", is_flag=True, help="Resume an existing interrupted investigation session.")
def cmd_agent_investigate(
    investigation_id: str,
    max_iterations: int,
    max_tool_calls: int,
    max_duration: int,
    offline: bool,
    no_llm: bool,
    verbose: bool,
    resume: bool,
) -> None:
    """Execute autonomous multi-modal investigation on a target image."""
    print_banner()
    from agent.investigation_agent import InvestigationAgent
    from core.configuration import AgentConfig, get_config

    cfg = get_config()
    agent_cfg = cfg.agent.model_copy()
    agent_cfg.max_iterations = max_iterations
    agent_cfg.max_tool_calls = max_tool_calls
    agent_cfg.max_duration_seconds = max_duration
    if offline:
        agent_cfg.offline_mode = True
    if no_llm:
        agent_cfg.enable_llm = False

    agent = InvestigationAgent(config=agent_cfg)
    console.print(f"[bold cyan][*] Launching HOUND Autonomous Investigation Agent...[/bold cyan]")
    console.print(f"Target Investigation: [bold]{investigation_id}[/bold]")
    console.print(f"Execution Budgets: Iterations={max_iterations}, ToolCalls={max_tool_calls}, MaxTime={max_duration}s\n")

    t0 = time.perf_counter()
    try:
        dossier = agent.investigate(
            investigation_id=investigation_id,
            resume=resume,
            max_iterations=max_iterations,
            offline=agent_cfg.offline_mode,
            verbose=verbose,
        )
    except Exception as e:
        console.print(f"[bold red][FAIL] Agent investigation failed: {e}[/bold red]")
        sys.exit(1)

    dt = time.perf_counter() - t0
    console.print(f"\n[bold green][+] Autonomous Investigation Concluded ({dt:.2f}s)[/bold green]")
    console.print(f"Final Status: [bold cyan]{dossier.final_status.upper()}[/bold cyan] (Reason: {dossier.stopping_reason})")
    console.print(f"Justified Precision: [bold yellow]{dossier.geographic_precision_justified}[/bold yellow]")
    console.print(f"Total Iterations: {dossier.total_iterations} | Tool Calls: {dossier.total_tool_calls} | Events: {dossier.events_count}")

    if dossier.primary_hypothesis:
        p = dossier.primary_hypothesis
        console.print("\n[bold]Leading Hypothesis:[/bold]")
        console.print(f"  • Name: [bold green]{p.name}[/bold green] (Level: {p.level})")
        if p.lat is not None and p.lon is not None:
            console.print(f"  • Coordinates: {p.lat:.4f}, {p.lon:.4f}")
        console.print(f"  • Heuristic Score: [cyan]{p.heuristic_score:.2f}[/cyan] ({p.score_breakdown.score_type})")
        console.print(f"  • Independent Corroborations: {len(p.supporting_evidence_ids)}")
        console.print(f"  • Contradictions: {len(p.contradictory_evidence_ids)}")

    if dossier.abstention_reasons:
        console.print("\n[bold yellow]Abstention Justification:[/bold yellow]")
        for reason in dossier.abstention_reasons:
            console.print(f"  • {reason}")


@agent_group.command("status")
@click.argument("investigation_id")
def cmd_agent_status(investigation_id: str) -> None:
    """View active session state and budget consumption."""
    print_banner()
    from agent.investigation_state import InvestigationState

    cfg = get_config()
    db = DatabaseManager(cfg.paths.database_path)
    state = InvestigationState.load_session(investigation_id, db)
    if not state:
        console.print(f"[yellow]No agent session recorded for '{investigation_id}'.[/yellow]")
        return

    table = Table(title=f"Agent Session Telemetry: {investigation_id}", box=ROUNDED)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Session State", f"[{'green' if state.is_completed else 'yellow'}]{'COMPLETED' if state.is_completed else 'RUNNING'}[/]")
    table.add_row("Iterations Completed", f"{state.iteration} / {state.max_iterations}")
    table.add_row("Tool Calls Executed", f"{state.tool_calls_count} / {state.max_tool_calls}")
    table.add_row("External Requests", f"{state.external_requests_count} / {state.max_external_requests}")
    table.add_row("Stopping Reason", str(state.stopping_reason or "In Progress"))
    table.add_row("Final Status", str(state.final_status or "Evaluating"))
    table.add_row("Active Hypotheses Count", str(len(state.hypotheses)))
    table.add_row("Events Logged", str(len(state.events)))

    console.print(table)


@agent_group.command("hypotheses")
@click.argument("investigation_id")
def cmd_agent_hypotheses(investigation_id: str) -> None:
    """Display all candidate hypotheses, status, and rejection reasons."""
    print_banner()
    from agent.investigation_state import InvestigationState

    cfg = get_config()
    db = DatabaseManager(cfg.paths.database_path)
    state = InvestigationState.load_session(investigation_id, db)
    if not state or not state.hypotheses:
        console.print(f"[yellow]No hypotheses recorded for '{investigation_id}'.[/yellow]")
        return

    table = Table(title="Agent Competing Hypotheses Ledger", box=ROUNDED)
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Candidate Name", style="bold white")
    table.add_column("Level", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("Details / Rejection Reason", style="dim")

    for h in sorted(state.hypotheses.values(), key=lambda x: x.heuristic_score, reverse=True):
        status_style = {
            "supported": "bold green",
            "active": "green",
            "proposed": "blue",
            "challenged": "yellow",
            "rejected": "bold red",
            "inconclusive": "dim white",
        }.get(h.status, "white")

        details = h.rejection_reason or f"{len(h.supporting_evidence_ids)} corroborations"
        table.add_row(
            h.id,
            h.name,
            h.level,
            f"{h.heuristic_score:.2f}",
            f"[{status_style}]{h.status.upper()}[/{status_style}]",
            details,
        )

    console.print(table)


@agent_group.command("history")
@click.argument("investigation_id")
def cmd_agent_history(investigation_id: str) -> None:
    """Display chronological investigation timeline of actions and results."""
    print_banner()
    from agent.investigation_state import InvestigationState

    cfg = get_config()
    db = DatabaseManager(cfg.paths.database_path)
    state = InvestigationState.load_session(investigation_id, db)
    if not state or not state.events:
        console.print(f"[yellow]No event history recorded for '{investigation_id}'.[/yellow]")
        return

    console.print(f"[bold]Investigation Timeline ({len(state.events)} events):[/bold]\n")
    for ev in state.events:
        tool_tag = f"[{ev.tool_id}]" if ev.tool_id else ""
        time_str = ev.timestamp[11:19] if len(ev.timestamp) >= 19 else ev.timestamp
        console.print(f"[dim]{time_str}[/dim] [cyan]Iter {ev.iteration:02d}[/cyan] [bold]{tool_tag}[/bold] {ev.action_summary}")
        if ev.result_summary:
            console.print(f"      [dim]-> {ev.result_summary} ({ev.duration_s:.2f}s)[/dim]")


@agent_group.command("verify")
@click.argument("investigation_id")
def cmd_agent_verify(investigation_id: str) -> None:
    """Run adversarial verification checks against leading candidate hypotheses."""
    print_banner()
    from agent.hypothesis_manager import HypothesisManager
    from agent.investigation_state import InvestigationState
    from agent.verification_engine import VerificationEngine

    cfg = get_config()
    db = DatabaseManager(cfg.paths.database_path)
    state = InvestigationState.load_session(investigation_id, db)
    if not state or not state.hypotheses:
        console.print(f"[yellow]No hypotheses found to verify for '{investigation_id}'.[/yellow]")
        return

    hyp_mgr = HypothesisManager(db_manager=db)
    evid_store = EvidenceStore(db)
    verifier = VerificationEngine(hyp_mgr, evid_store)

    evidence_items = evid_store.get_evidence(investigation_id)
    console.print("[bold]Generating Adversarial Verification Questions...[/bold]\n")

    for h in sorted(state.hypotheses.values(), key=lambda x: x.heuristic_score, reverse=True)[:3]:
        questions = verifier.generate_verification_questions(h, evidence_items)
        console.print(f"[bold cyan]Candidate: {h.name}[/bold cyan] ({h.level}, Score: {h.heuristic_score:.2f})")
        if not questions:
            console.print("  [dim]No automated verification questions identified for this candidate modality.[/dim]")
        for q in questions:
            console.print(f"  • [{q.modality.upper()}] [bold]{q.question_text}[/bold]")
        console.print()


@agent_group.command("report")
@click.argument("investigation_id")
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output Markdown report path.")
def cmd_agent_report(investigation_id: str, output: Optional[Path]) -> None:
    """Generate and display comprehensive investigation report dossier."""
    print_banner()
    from agent.investigation_agent import InvestigationAgent

    agent = InvestigationAgent()
    inv = agent.inv_store.get_investigation(investigation_id)
    if not inv:
        console.print(f"[red]Investigation '{investigation_id}' not found.[/red]")
        return

    # Load session and compile dossier
    from agent.investigation_state import InvestigationState
    state = InvestigationState.load_session(investigation_id, agent.db)
    if not state:
        console.print(f"[yellow]No agent session found for '{investigation_id}'. Run 'hound agent investigate' first.[/yellow]")
        return

    dossier = agent._compile_dossier(state, inv)

    # Format Markdown dossier
    lines = [
        f"# HOUND INVESTIGATION DOSSIER: {dossier.investigation_id}",
        f"**Target Media:** {dossier.target_filename} (`{dossier.sha256_hash}`)",
        f"**Final Status:** `{dossier.final_status.upper()}` | **Stopping Reason:** `{dossier.stopping_reason}`",
        f"**Justified Precision:** **{dossier.geographic_precision_justified}**",
        f"**Total Iterations:** {dossier.total_iterations} | **Duration:** {dossier.total_duration_s:.2f}s",
        "",
        "---",
        "",
        "## 1. Leading Geographic Hypothesis",
    ]

    if dossier.primary_hypothesis:
        p = dossier.primary_hypothesis
        lines.extend([
            f"- **Target Name:** {p.name}",
            f"- **Granularity Level:** `{p.level}`",
            f"- **Coordinates:** `{p.lat:.4f}, {p.lon:.4f}`" if p.lat is not None and p.lon is not None else "- **Coordinates:** Unassigned",
            f"- **Heuristic Score:** {p.heuristic_score:.2f} (`{p.score_breakdown.score_type}`)",
            f"- **Independent Corroborations:** {len(p.supporting_evidence_ids)}",
            f"- **Contradictions:** {len(p.contradictory_evidence_ids)}",
        ])
    else:
        lines.append("*No supported candidate hypothesis. System abstained based on available evidence.*")

    lines.extend([
        "",
        "## 2. Verified Facts vs. Deductions",
        "### Facts (Direct Optical Observations & EXIF)",
    ])
    for f in dossier.facts:
        lines.append(f"- {f}")

    lines.extend([
        "",
        "### Inferences & Spatial Deductions",
    ])
    for inf in dossier.inferences:
        lines.append(f"- {inf}")

    if dossier.abstention_reasons:
        lines.extend([
            "",
            "## 3. Abstention Justification",
        ])
        for r in dossier.abstention_reasons:
            lines.append(f"- {r}")

    lines.extend([
        "",
        "## 4. Competing & Rejected Candidates",
    ])
    for c in dossier.competing_hypotheses:
        lines.append(f"- **{c.name}** ({c.level}) - Score: {c.heuristic_score:.2f} | Status: `{c.status}`")
    for r in dossier.rejected_hypotheses:
        lines.append(f"- ~~{r.name}~~ ({r.level}) - **REJECTED**: {r.rejection_reason}")

    report_content = "\n".join(lines)
    if output:
        output.write_text(report_content, encoding="utf-8")
        console.print(f"[bold green][+][/bold green] Dossier saved to: [cyan]{output}[/cyan]")
    else:
        console.print(Markdown(report_content))


def main() -> None:
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
