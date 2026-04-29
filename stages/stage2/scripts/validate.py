#!/usr/bin/env python

"""
Run validation for LTXV models using configuration from YAML files.

This script loads the configuration and runs the validation process (sample generation)
as defined in the trainer.
"""

from pathlib import Path

import typer
import yaml
import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ltxv_trainer.config import LtxvTrainerConfig
from ltxv_trainer.trainer import LtxvTrainer

console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Run validation for LTXV models using configuration from YAML files.",
)


@app.command()
def main(
    config_path: str = typer.Argument(..., help="Path to YAML configuration file"),
    remove_last_frames: int = typer.Option(
        None,
        "--remove-last-frames",
        help="Number of frames to remove from the end of the generated video. Overrides config value if set.",
    ),
    input_folder: str = typer.Option(
        None,
        "--input-folder",
        "-i",
        help="Folder containing reference video mp4 files to process. Overrides reference_videos in config.",
    ),
    noisemap_videos_folder: str = typer.Option(
        None,
        "--noisemap-videos-folder",
        "-n",
        help="Folder containing noisemap video mp4 files. Overrides noisemap_videos in config.",
    ),
    output_folder: str = typer.Option(
        None,
        "--output-folder",
        "-o",
        help="Output folder for generated videos. Overrides output_dir in config.",
    ),
) -> None:
    """Run validation using the provided configuration file."""
    # Load the configuration from the YAML file
    config_path = Path(config_path)
    if not config_path.exists():
        typer.echo(f"Error: Configuration file {config_path} does not exist.")
        raise typer.Exit(code=1)

    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)

    # Convert the loaded data to the LtxvTrainerConfig object
    try:
        trainer_config = LtxvTrainerConfig(**config_data)

        # Override remove_last_n_frames if provided via command line
        if remove_last_frames is not None:
             trainer_config.validation.remove_last_n_frames = remove_last_frames
             console.print(f"[cyan]ℹ️  Remove last frames set to: {remove_last_frames}[/cyan]")

        # Override reference_videos if input_folder is provided
        if input_folder is not None:
            input_path = Path(input_folder)
            if not input_path.exists():
                typer.echo(f"Error: Input folder {input_path} does not exist.")
                raise typer.Exit(code=1)

            # Find all mp4 files in the input folder
            mp4_files = sorted(input_path.glob("*.mp4"))
            if not mp4_files:
                typer.echo(f"Error: No mp4 files found in {input_path}")
                raise typer.Exit(code=1)

            # Update reference_videos with all found mp4 files
            trainer_config.validation.reference_videos = [str(f) for f in mp4_files]
            # Update prompts to match the number of videos (use the first prompt for all)
            if trainer_config.validation.prompts:
                first_prompt = trainer_config.validation.prompts[0]
                trainer_config.validation.prompts = [first_prompt] * len(mp4_files)

            console.print(f"[cyan]ℹ️  Input folder: {input_path}[/cyan]")
            console.print(f"[cyan]ℹ️  Found {len(mp4_files)} reference video file(s) to process[/cyan]")

        # Override noisemap_videos if noisemap_videos_folder is provided
        if noisemap_videos_folder is not None:
            noisemap_path = Path(noisemap_videos_folder)
            if not noisemap_path.exists():
                typer.echo(f"Error: Noisemap videos folder {noisemap_path} does not exist.")
                raise typer.Exit(code=1)

            # Find all mp4 files in the noisemap folder
            noisemap_mp4_files = sorted(noisemap_path.glob("*.mp4"))
            if not noisemap_mp4_files:
                typer.echo(f"Error: No mp4 files found in {noisemap_path}")
                raise typer.Exit(code=1)

            # If input_folder was also provided, match by filename to ensure correct pairing
            if input_folder is not None:
                # Create a mapping from filename to full path for noisemap videos
                noisemap_dict = {f.name: str(f) for f in noisemap_mp4_files}

                # Match noisemap videos to reference videos by filename
                matched_noisemap_videos = []
                missing_noisemap = []

                for ref_video_path in trainer_config.validation.reference_videos:
                    ref_filename = Path(ref_video_path).name
                    if ref_filename in noisemap_dict:
                        matched_noisemap_videos.append(noisemap_dict[ref_filename])
                    else:
                        missing_noisemap.append(ref_filename)
                        typer.echo(f"Warning: No matching noisemap video found for {ref_filename}")

                if missing_noisemap:
                    console.print(f"[yellow]⚠️  Warning: {len(missing_noisemap)} reference video(s) have no matching noisemap video[/yellow]")
                    typer.echo(f"Error: Cannot proceed with mismatched videos. Missing noisemap videos for: {', '.join(missing_noisemap[:5])}{'...' if len(missing_noisemap) > 5 else ''}")
                    raise typer.Exit(code=1)

                trainer_config.validation.noisemap_videos = matched_noisemap_videos
                console.print(f"[cyan]ℹ️  Noisemap videos folder: {noisemap_path}[/cyan]")
                console.print(f"[cyan]ℹ️  Matched {len(matched_noisemap_videos)} noisemap video(s) with reference videos[/cyan]")
            else:
                # No input_folder provided, just use all noisemap videos as-is
                trainer_config.validation.noisemap_videos = [str(f) for f in noisemap_mp4_files]
                console.print(f"[cyan]ℹ️  Noisemap videos folder: {noisemap_path}[/cyan]")
                console.print(f"[cyan]ℹ️  Found {len(noisemap_mp4_files)} noisemap video file(s)[/cyan]")

        # Override output_dir if output_folder is provided
        if output_folder is not None:
            trainer_config.output_dir = output_folder
            console.print(f"[cyan]ℹ️  Output folder: {output_folder}[/cyan]")

    except Exception as e:
        typer.echo(f"Error: Invalid configuration data: {e}")
        raise typer.Exit(code=1) from e

    # Print validation summary BEFORE loading models
    num_videos = len(trainer_config.validation.reference_videos) if trainer_config.validation.reference_videos else len(trainer_config.validation.prompts)
    console.print("\n" + "="*60)
    console.print("[bold green]🚀 Starting Validation[/bold green]")
    console.print("="*60)
    console.print(f"[yellow]📝 Configuration:[/yellow]")
    console.print(f"   • Config file: {config_path}")
    console.print(f"   • Number of videos to generate: [bold]{num_videos}[/bold]")
    console.print(f"   • Output directory: {trainer_config.output_dir}/samples")
    console.print(f"   • Video dimensions: {trainer_config.validation.video_dims[0]}x{trainer_config.validation.video_dims[1]}x{trainer_config.validation.video_dims[2]} (W×H×F)")
    console.print(f"   • Inference steps: {trainer_config.validation.inference_steps}")
    console.print(f"   • Guidance scale: {trainer_config.validation.guidance_scale}")
    if trainer_config.validation.remove_last_n_frames:
        console.print(f"   • Remove last frames: {trainer_config.validation.remove_last_n_frames}")
    if hasattr(trainer_config.validation, 'noisemap_videos') and trainer_config.validation.noisemap_videos:
        console.print(f"   • Noisemap videos: [bold]{len(trainer_config.validation.noisemap_videos)}[/bold] files")
    if hasattr(trainer_config.validation, 'auxiliary_latent_path') and trainer_config.validation.auxiliary_latent_path:
        console.print(f"   • Auxiliary latent A: [bold]{trainer_config.validation.auxiliary_latent_path}[/bold]")
    if hasattr(trainer_config.validation, 'auxiliary_latent_b_path') and trainer_config.validation.auxiliary_latent_b_path:
        console.print(f"   • Auxiliary latent B: [bold]{trainer_config.validation.auxiliary_latent_b_path}[/bold]")
    if hasattr(trainer_config.validation, 'noisemap_blend_weights') and trainer_config.validation.noisemap_blend_weights:
        has_latent_b = hasattr(trainer_config.validation, 'auxiliary_latent_b_path') and trainer_config.validation.auxiliary_latent_b_path
        blend_mode = "3-way" if has_latent_b else "2-way"
        console.print(f"   • Blend weights: [bold]Enabled ({blend_mode})[/bold] ({len(trainer_config.validation.noisemap_blend_weights)} steps)")
    console.print("="*60 + "\n")

    # Initialize the trainer (which loads models)
    # Note: We initialize LtxvTrainer to reuse its setup and model loading logic
    console.print("[cyan]Loading models...[/cyan]")
    trainer = LtxvTrainer(trainer_config)
    
    # We need to manually set up some things that are usually set up in train()
    # or ensure they are ready for _sample_videos
    
    # Ensure models are on the correct device if not handled by accelerator prepare yet
    # LtxvTrainer.__init__ calls _setup_accelerator and _load_models.
    # _load_models leaves models on CPU (except maybe 8bit text encoder) and frozen.
    # _sample_videos handles moving VAE/TextEncoder to device.
    # But Transformer might need to be moved if not prepared.
    
    # In train(), prepare_models_for_training calls accelerator.prepare.
    # We should probably call that to ensure everything is set up correctly for inference too,
    # or manually move the transformer if we just want to run validation.
    
    # Let's use the trainer's internal methods to prepare minimal required state
    # The trainer is already initialized, so models are loaded.
    
    # If using LoRA, we need to make sure LoRA weights are loaded/initialized
    if trainer_config.model.training_mode == "lora":
         # In train(), _init_lora_weights is called if not loading checkpoint.
         # But here we probably want to validate a specific checkpoint or the base model + lora config.
         # If load_checkpoint is set in config, _load_checkpoint was called in __init__.
         pass

    # Create a progress bar for sampling
    sample_progress = Progress(
        TextColumn("Sampling validation videos"),
        MofNCompleteColumn(),
        BarColumn(bar_width=40, style="blue"),
        TimeElapsedColumn(),
        TextColumn("ETA:"),
        TimeRemainingColumn(compact=True),
    )
    
    console.print("🚀 Starting validation...")
    
    with sample_progress:
        # We need to set a dummy global step for filename generation if not set
        if trainer._global_step == -1:
             trainer._global_step = 0
             
        # Call the private method _sample_videos
        # Note: This is a bit of a hack accessing private method, but avoids code duplication
        video_paths = trainer._sample_videos(sample_progress)
        
    if video_paths:
        console.print(f"Validation completed. Generated {len(video_paths)} videos.")
        for path in video_paths:
            console.print(f"  - {path}")
    else:
        console.print("⚠️ No videos generated. Check validation configuration.")


if __name__ == "__main__":
    app()
