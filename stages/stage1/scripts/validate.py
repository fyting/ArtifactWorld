#!/usr/bin/env python

"""
Run validation for LTXV models using configuration from YAML files.

This script loads the configuration and runs the validation process (sample generation)
as defined in the trainer.
"""

from pathlib import Path
import time

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
        help="Folder containing mp4 files to process. Overrides reference_videos in config.",
    ),
    output_folder: str = typer.Option(
        None,
        "--output-folder",
        "-o",
        help="Output folder for generated videos. Overrides output_dir in config.",
    ),
) -> None:
    """Run validation using the provided configuration file."""
    # Record start time
    script_start_time = time.time()
    console.print(f"[yellow]⏱️  Script execution started at: {time.strftime('%Y-%m-%d %H:%M:%S')}[/yellow]")

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
            console.print(f"[cyan]ℹ️  Found {len(mp4_files)} mp4 file(s) to process[/cyan]")

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
    console.print("="*60 + "\n")

    # Initialize the trainer (which loads models)
    # Note: We initialize LtxvTrainer to reuse its setup and model loading logic
    console.print("[cyan]Loading models...[/cyan]")
    model_load_start = time.time()
    trainer = LtxvTrainer(trainer_config)
    model_load_time = time.time() - model_load_start
    console.print(f"[yellow]⏱️  Model loading time: {model_load_time:.2f}s[/yellow]")
    
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

    init_time = time.time() - script_start_time
    console.print(f"[yellow]⏱️  Total initialization time: {init_time:.2f}s (from script start)[/yellow]")

    inference_start_time = time.time()
    with sample_progress:
        # We need to set a dummy global step for filename generation if not set
        if trainer._global_step == -1:
             trainer._global_step = 0

        # Call the private method _sample_videos
        # Note: This is a bit of a hack accessing private method, but avoids code duplication
        video_paths = trainer._sample_videos(sample_progress)

    inference_time = time.time() - inference_start_time
    total_time = time.time() - script_start_time

    if video_paths:
        console.print(f"Validation completed. Generated {len(video_paths)} videos.")
        console.print(f"[yellow]⏱️  Inference time: {inference_time:.2f}s[/yellow]")
        console.print(f"[yellow]⏱️  Average time per video: {inference_time/len(video_paths):.2f}s[/yellow]")
        console.print(f"[yellow]⏱️  Total time: {total_time:.2f}s[/yellow]")
        for path in video_paths:
            console.print(f"  - {path}")
    else:
        console.print("⚠️ No videos generated. Check validation configuration.")


if __name__ == "__main__":
    app()
