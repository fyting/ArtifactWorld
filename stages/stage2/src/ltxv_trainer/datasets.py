# Adapted from https://github.com/a-r-r-o-w/finetrainers/blob/main/finetrainers/dataset.py

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ltxv_trainer import logger

# Constants for precomputed data directories
PRECOMPUTED_DIR_NAME = ".precomputed"


class DummyDataset(Dataset):
    """Produce random latents and prompt embeddings. For minimal demonstration and benchmarking purposes"""

    def __init__(
        self,
        width: int = 1024,
        height: int = 1024,
        num_frames: int = 25,
        fps: int = 24,
        dataset_length: int = 200,
        latent_dim: int = 128,
        latent_spatial_compression_ratio: int = 32,
        latent_temporal_compression_ratio: int = 8,
        prompt_embed_dim: int = 4096,
        prompt_sequence_length: int = 256,
    ) -> None:
        if width % 32 != 0:
            raise ValueError(f"Width must be divisible by 32, got {width=}")

        if height % 32 != 0:
            raise ValueError(f"Height must be divisible by 32, got {height=}")

        if num_frames % 8 != 1:
            raise ValueError(f"Number of frames must have a remainder of 1 when divided by 8, got {num_frames=}")

        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.fps = fps
        self.dataset_length = dataset_length
        self.latent_dim = latent_dim
        self.num_latent_frames = (num_frames - 1) // latent_temporal_compression_ratio + 1
        self.latent_height = height // latent_spatial_compression_ratio
        self.latent_width = width // latent_spatial_compression_ratio
        self.latent_sequence_length = self.num_latent_frames * self.latent_height * self.latent_width
        self.prompt_embed_dim = prompt_embed_dim
        self.prompt_sequence_length = prompt_sequence_length

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> dict[str, dict[str, Tensor]]:
        return {
            "latent_conditions": {
                "latents": torch.randn(1, self.latent_sequence_length, self.latent_dim),  # random video latents
                "num_frames": self.num_latent_frames,
                "height": self.latent_height,
                "width": self.latent_width,
                "fps": self.fps,
            },
            "text_conditions": {
                "prompt_embeds": torch.randn(
                    self.prompt_sequence_length,
                    self.prompt_embed_dim,
                ),  # random text embeddings
                "prompt_attention_mask": torch.ones(
                    self.prompt_sequence_length,
                    dtype=torch.bool,
                ),  # random attention mask
            },
        }


class PrecomputedDataset(Dataset):
    def __init__(self, data_root: str, data_sources: dict[str, str] | list[str] | None = None) -> None:
        """
        Generic dataset for loading precomputed data from multiple sources.

        Args:
            data_root: Root directory containing preprocessed data
            data_sources: Either:
                         - Dict mapping directory names to output keys
                         - List of directory names (keys will equal values)
                         - None (defaults to ["latents", "conditions"])

        Example:
            # Standard mode (list)
            dataset = PrecomputedDataset("data/", ["latents", "conditions"])

            # Standard mode (dict)
            dataset = PrecomputedDataset("data/", {"latents": "latent_conditions", "conditions": "text_conditions"})

            # IC-LoRA mode
            dataset = PrecomputedDataset("data/", ["latents", "conditions", "ref_latents"])
        """
        super().__init__()

        self.data_root = self._setup_data_root(data_root)
        self.data_sources = self._normalize_data_sources(data_sources)
        self.source_paths = self._setup_source_paths()
        self.sample_files = self._discover_samples()
        self._validate_setup()

    @staticmethod
    def _setup_data_root(data_root: str) -> Path:
        """Setup and validate the data root directory."""
        data_root = Path(data_root)

        if not data_root.exists():
            raise FileNotFoundError(f"Data root directory does not exist: {data_root}")

        # If the given path is the dataset root, use the precomputed sub-directory
        if (data_root / PRECOMPUTED_DIR_NAME).exists():
            data_root = data_root / PRECOMPUTED_DIR_NAME

        return data_root

    @staticmethod
    def _normalize_data_sources(data_sources: dict[str, str] | list[str] | None) -> dict[str, str]:
        """Normalize data_sources input to a consistent dict format."""
        if data_sources is None:
            # Default sources
            return {"latents": "latent_conditions", "conditions": "text_conditions"}
        elif isinstance(data_sources, list):
            # Convert list to dict where keys equal values
            return {source: source for source in data_sources}
        elif isinstance(data_sources, dict):
            return data_sources.copy()
        else:
            raise TypeError(f"data_sources must be dict, list, or None, got {type(data_sources)}")

    def _setup_source_paths(self) -> dict[str, Path]:
        """Map data source names to their actual directory paths."""
        source_paths = {}

        for dir_name in self.data_sources:
            source_path = self.data_root / dir_name
            source_paths[dir_name] = source_path

            # Check that all sources exist.
            if not source_path.exists():
                raise FileNotFoundError(f"Required {dir_name} directory does not exist: {source_path}")

        return source_paths

    @staticmethod
    def _find_all_pt_files(root_path: Path) -> list[Path]:
        """Find all .pt files recursively, following symlinks.

        Python's glob/rglob doesn't follow symlinks by default, so we need
        to manually handle symlinked directories.

        Args:
            root_path: Root directory to search

        Returns:
            List of paths to all .pt files found
        """
        all_files = []

        def _recursive_search(path: Path, visited_dirs: set[Path]) -> None:
            """Recursively search for .pt files, following symlinks."""
            # Resolve path to handle symlinks and get canonical path
            try:
                resolved_path = path.resolve()
            except (OSError, RuntimeError):
                # Skip paths that can't be resolved (broken symlinks, etc.)
                logger.warning(f"Could not resolve path: {path}")
                return

            # Avoid infinite loops from circular symlinks
            if resolved_path in visited_dirs:
                return
            visited_dirs.add(resolved_path)

            # Check if path exists and is accessible
            if not path.exists():
                return

            # If it's a file, check if it's a .pt file
            if path.is_file():
                if path.suffix == ".pt":
                    all_files.append(path)
                return

            # If it's a directory, recurse into it
            if path.is_dir():
                try:
                    for item in path.iterdir():
                        _recursive_search(item, visited_dirs)
                except PermissionError:
                    logger.warning(f"Permission denied accessing: {path}")

        _recursive_search(root_path, set())
        return all_files

    def _discover_samples(self) -> dict[str, list[Path]]:
        """Discover all valid sample files across all data sources.

        This method handles flexible directory structures by matching files based on
        their basename (filename without path) rather than requiring identical relative paths.
        """
        # Step 1: Build a mapping of filename -> full_path for each data source
        file_mappings = {}  # {dir_name: {filename: full_path}}

        for dir_name, source_path in self.source_paths.items():
            file_mappings[dir_name] = {}
            all_files = self._find_all_pt_files(source_path)

            if not all_files:
                logger.warning(f"No .pt files found in {source_path}")
                continue

            for file_path in all_files:
                filename = file_path.name  # Just the filename (e.g., "abc123.pt")
                if filename in file_mappings[dir_name]:
                    logger.warning(
                        f"Duplicate filename '{filename}' found in {dir_name}: "
                        f"{file_mappings[dir_name][filename]} and {file_path}"
                    )
                file_mappings[dir_name][filename] = file_path

            logger.debug(f"Found {len(file_mappings[dir_name])} files in {dir_name}")

        # Step 2: Find common filenames across all data sources
        # Use latents as reference if available, otherwise use first source
        reference_dir = "latents" if "latents" in self.data_sources else next(iter(self.data_sources.keys()))

        if reference_dir not in file_mappings or not file_mappings[reference_dir]:
            raise ValueError(f"No data files found in reference source: {reference_dir}")

        reference_filenames = set(file_mappings[reference_dir].keys())
        logger.debug(f"Reference source '{reference_dir}' has {len(reference_filenames)} files")

        # Step 3: Find filenames that exist in ALL data sources
        common_filenames = reference_filenames.copy()
        for dir_name in self.data_sources.keys():
            if dir_name not in file_mappings:
                raise ValueError(f"No files found in required data source: {dir_name}")

            source_filenames = set(file_mappings[dir_name].keys())
            common_filenames &= source_filenames
            logger.debug(
                f"After checking '{dir_name}': {len(common_filenames)} common files remain "
                f"(source has {len(source_filenames)} files)"
            )

        if not common_filenames:
            raise ValueError(
                "No common filenames found across all data sources. "
                "Please ensure each sample has corresponding files in all required directories."
            )

        logger.info(f"Found {len(common_filenames)} valid samples across all data sources")

        # Step 4: Build the sample files dict with relative paths
        sample_files = {output_key: [] for output_key in self.data_sources.values()}

        for filename in sorted(common_filenames):  # Sort for deterministic order
            for dir_name, output_key in self.data_sources.items():
                full_path = file_mappings[dir_name][filename]
                rel_path = full_path.relative_to(self.source_paths[dir_name])
                sample_files[output_key].append(rel_path)

        return sample_files


    def _validate_setup(self) -> None:
        """Validate that the dataset setup is correct."""
        if not self.sample_files:
            raise ValueError("No valid samples found - all data sources must have matching files")

        # Verify all output keys have the same number of samples
        sample_counts = {key: len(files) for key, files in self.sample_files.items()}
        if len(set(sample_counts.values())) > 1:
            raise ValueError(f"Mismatched sample counts across sources: {sample_counts}")

    def __len__(self) -> int:
        # Use the first output key as reference count
        first_key = next(iter(self.sample_files.keys()))
        return len(self.sample_files[first_key])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        result = {}

        for dir_name, output_key in self.data_sources.items():
            source_path = self.source_paths[dir_name]
            file_rel_path = self.sample_files[output_key][index]
            file_path = source_path / file_rel_path

            try:
                data = torch.load(file_path, map_location="cpu", weights_only=True)
                result[output_key] = data
            except Exception as e:
                raise RuntimeError(f"Failed to load {output_key} from {file_path}: {e}") from e

        # Add index for debugging
        result["idx"] = index
        return result
