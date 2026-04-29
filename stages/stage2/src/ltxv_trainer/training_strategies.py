"""Training strategies for different conditioning modes.

This module implements the Strategy Pattern to handle different training modes:
- Standard training (no conditioning)
- Reference video training (IC-LoRA mode)

Each strategy encapsulates the specific logic for preparing batches, model inputs, and loss computation.
"""

import random
from abc import ABC, abstractmethod
from typing import Any

import torch
from pydantic import BaseModel, computed_field
from torch import Tensor

from ltxv_trainer import logger
from ltxv_trainer.config import ConditioningConfig
from ltxv_trainer.ltxv_utils import get_rope_scale_factors, prepare_video_coordinates
from ltxv_trainer.timestep_samplers import TimestepSampler

DEFAULT_FPS = 24  # Default frames per second for video missing in the FPS metadata


class TrainingBatch(BaseModel):
    """Container for prepared training data.

    This model holds all the prepared data needed for a training step,
    organized in a way that's agnostic to the specific training strategy.
    """

    # Core latent data
    latents: Tensor  # The main latent input to the transformer
    targets: Tensor  # The target values for loss computation

    # Text conditioning
    prompt_embeds: Tensor  # Text embeddings
    prompt_attention_mask: Tensor  # Attention mask for text

    # Timestep information
    timesteps: Tensor  # Timestep values for the transformer
    sigmas: Tensor  # Noise schedule values

    # Conditioning information
    conditioning_mask: Tensor  # Boolean mask: True = conditioning token, False = target token

    # Video metadata
    num_frames: int  # Number of frames in the video
    height: int  # Height of the video latents
    width: int  # Width of the video latents
    fps: float  # Frames per second

    # Model input parameters
    rope_interpolation_scale: list[float]  # Scaling factors for positional embeddings
    video_coords: Tensor | None = None  # Optional explicit video coordinates

    @computed_field
    @property
    def batch_size(self) -> int:
        """Compute batch size from latents tensor."""
        return self.latents.shape[0]

    @computed_field
    @property
    def sequence_length(self) -> int:
        """Compute sequence length from latents tensor."""
        return self.latents.shape[1]

    model_config = {"arbitrary_types_allowed": True}  # Allow torch.Tensor type


class TrainingStrategy(ABC):
    """Abstract base class for training strategies.

    Each strategy encapsulates the logic for a specific training mode,
    handling batch preparation, model input preparation, and loss computation.
    """

    def __init__(self, conditioning_config: ConditioningConfig):
        """Initialize strategy with conditioning configuration.

        Args:
            conditioning_config: Configuration for conditioning behavior
        """
        self.conditioning_config = conditioning_config

    @abstractmethod
    def get_data_sources(self) -> list[str] | dict[str, str]:
        """Get the required data sources for this training strategy.

        Returns:
            Dictionary mapping data directory names to output keys for the dataset
        """

    @abstractmethod
    def prepare_batch(self, batch: dict[str, Any], timestep_sampler: TimestepSampler) -> TrainingBatch:
        """Prepare a raw data batch for training.

        Args:
            batch: Raw batch data from the dataset
            timestep_sampler: Sampler for generating timesteps and noise

        Returns:
            Prepared training batch with all necessary data
        """

    def _create_timesteps_from_conditioning_mask(
        self, conditioning_mask: Tensor, sampled_timestep_values: Tensor
    ) -> Tensor:
        """Create timesteps based on conditioning mask.

        Args:
            conditioning_mask: Boolean mask of shape (batch_size, sequence_length),
            where True = conditioning, False = target.
            sampled_timestep_values: Sampled timestep values for target tokens of shape (batch_size,)

        Returns:
            Timesteps tensor with 0 for conditioning tokens, sampled values for target tokens
        """
        # Expand sampled values to match conditioning mask shape
        expanded_timesteps = sampled_timestep_values.unsqueeze(1).expand_as(conditioning_mask)

        # Use conditioning mask to select between 0 (conditioning) and sampled values (target)
        return torch.where(conditioning_mask, 0, expanded_timesteps)

    def _create_first_frame_conditioning_mask(
        self, batch_size: int, sequence_length: int, height: int, width: int, device: torch.device
    ) -> Tensor:
        """Create conditioning mask for first frame conditioning.

        Returns:
            Boolean mask where True indicates first frame tokens (if conditioning is enabled)
        """
        conditioning_mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool, device=device)

        if (
            self.conditioning_config.first_frame_conditioning_p > 0
            and random.random() < self.conditioning_config.first_frame_conditioning_p
        ):
            first_frame_end_idx = height * width
            if first_frame_end_idx < sequence_length:
                conditioning_mask[:, :first_frame_end_idx] = True

        return conditioning_mask

    def _create_last_frame_conditioning_mask(
        self, batch_size: int, sequence_length: int, height: int, width: int, device: torch.device
    ) -> Tensor:
        """Create conditioning mask for last frame conditioning.

        Returns:
            Boolean mask where True indicates last frame tokens (if conditioning is enabled)
        """
        conditioning_mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool, device=device)

        if (
            self.conditioning_config.last_frame_conditioning_p > 0
            and random.random() < self.conditioning_config.last_frame_conditioning_p
        ):
            frame_size = height * width
            last_frame_start_idx = sequence_length - frame_size
            if last_frame_start_idx >= 0:
                conditioning_mask[:, last_frame_start_idx:] = True

        return conditioning_mask

    def _create_middle_frame_conditioning_mask(
        self,
        batch_size: int,
        sequence_length: int,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> Tensor:
        """Create conditioning mask for random middle frame conditioning.

        Returns:
            Boolean mask where True indicates middle frame tokens (if conditioning is enabled)
        """
        conditioning_mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool, device=device)

        if (
            self.conditioning_config.middle_frame_conditioning_p > 0
            and random.random() < self.conditioning_config.middle_frame_conditioning_p
            and num_frames > 2
        ):
            frame_size = height * width
            # Pick a random frame index between 1 and num_frames - 2 (inclusive)
            middle_frame_idx = random.randint(1, num_frames - 2)
            start_idx = middle_frame_idx * frame_size
            end_idx = start_idx + frame_size

            if end_idx <= sequence_length:
                conditioning_mask[:, start_idx:end_idx] = True

        return conditioning_mask

    @staticmethod
    def prepare_model_inputs(batch: TrainingBatch) -> dict[str, Any]:
        """Prepare inputs for the transformer model.

        Args:
            batch: Prepared training data

        Returns:
            Dictionary of keyword arguments for the transformer forward call
        """

        return {
            "hidden_states": batch.latents,
            "encoder_hidden_states": batch.prompt_embeds,
            "timestep": batch.timesteps,
            "encoder_attention_mask": batch.prompt_attention_mask,
            "num_frames": batch.num_frames,
            "height": batch.height,
            "width": batch.width,
            "rope_interpolation_scale": batch.rope_interpolation_scale,
            "video_coords": batch.video_coords,
            "return_dict": False,
        }

    @abstractmethod
    def compute_loss(self, model_pred: Tensor, batch: TrainingBatch) -> Tensor:
        """Compute the training loss.

        Args:
            model_pred: Output from the transformer model
            batch: The prepared training data containing targets

        Returns:
            Scalar loss tensor
        """


class StandardTrainingStrategy(TrainingStrategy):
    """Standard training strategy without conditioning.

    This strategy implements regular video generation training where:
    - Only target latents are used (no reference videos)
    - Standard noise application and loss computation
    - Single video sequence length
    - Supports first frame conditioning
    """

    def __init__(self, conditioning_config: ConditioningConfig):
        """Initialize standard training strategy.

        Args:
            conditioning_config: Configuration for conditioning behavior
        """
        super().__init__(conditioning_config)

    def get_data_sources(self) -> list[str]:
        """Standard training requires latents and text conditions."""
        return ["latents", "conditions"]

    def prepare_batch(self, batch: dict[str, Any], timestep_sampler: TimestepSampler) -> TrainingBatch:
        """Prepare batch for standard training."""
        # Get pre-encoded latents
        latents = batch["latents"]
        target_latents = latents["latents"]

        # Note: Batch sizes > 1 are partially supported, assuming
        # num_frames, height, width, fps are the same for all batch elements.
        latent_frames = latents["num_frames"][0].item()
        latent_height = latents["height"][0].item()
        latent_width = latents["width"][0].item()

        # Handle FPS with backward compatibility for old preprocessed datasets
        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values found in the batch. Found: {fps.tolist()}, using the first one: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        # Get pre-encoded text conditions
        conditions = batch["conditions"]
        prompt_embeds = conditions["prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        # Create conditioning mask (only first frame conditioning for standard training)
        conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=target_latents.shape[0],
            sequence_length=target_latents.shape[1],
            height=latent_height,
            width=latent_width,
            device=target_latents.device,
        )
        last_frame_mask = self._create_last_frame_conditioning_mask(
            batch_size=target_latents.shape[0],
            sequence_length=target_latents.shape[1],
            height=latent_height,
            width=latent_width,
            device=target_latents.device,
        )
        conditioning_mask = conditioning_mask | last_frame_mask

        middle_frame_mask = self._create_middle_frame_conditioning_mask(
            batch_size=target_latents.shape[0],
            sequence_length=target_latents.shape[1],
            num_frames=latent_frames,
            height=latent_height,
            width=latent_width,
            device=target_latents.device,
        )
        conditioning_mask = conditioning_mask | middle_frame_mask

        # Create noise for the target latents
        sigmas = timestep_sampler.sample_for(target_latents)
        noise = torch.randn_like(target_latents, device=target_latents.device)

        # Apply noise only to non-conditioning tokens
        sigmas = sigmas.view(-1, 1, 1)
        noisy_latents = (1 - sigmas) * target_latents + sigmas * noise

        # For conditioning tokens, use clean latents instead of noisy ones
        conditioning_mask_expanded = conditioning_mask.unsqueeze(-1)  # (B, seq_len, 1)
        noisy_latents = torch.where(conditioning_mask_expanded, target_latents, noisy_latents)

        targets = noise - target_latents

        # Create timesteps based on conditioning mask
        sampled_timestep_values = torch.round(sigmas.squeeze(-1).squeeze(-1) * 1000.0).long()
        timesteps = self._create_timesteps_from_conditioning_mask(conditioning_mask, sampled_timestep_values)

        # Use existing utility function for ROPE scale factors
        rope_interpolation_scale_factors = get_rope_scale_factors(fps)

        return TrainingBatch(
            latents=noisy_latents,
            targets=targets,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            timesteps=timesteps,
            sigmas=sigmas,
            conditioning_mask=conditioning_mask,
            num_frames=latent_frames,
            height=latent_height,
            width=latent_width,
            fps=fps,
            rope_interpolation_scale=rope_interpolation_scale_factors,
            video_coords=None,
        )

    def compute_loss(self, model_pred: Tensor, batch: TrainingBatch) -> Tensor:
        """Compute masked MSE loss using conditioning mask."""
        loss = (model_pred - batch.targets).pow(2)

        # Create loss mask: exclude conditioning tokens
        loss_mask = (~batch.conditioning_mask.unsqueeze(-1)).float()

        # Apply original loss computation pattern
        loss = loss.mul(loss_mask).div(loss_mask.mean())
        return loss.mean()


class ReferenceVideoTrainingStrategy(TrainingStrategy):
    """Reference video training strategy for IC-LoRA.

    This strategy implements training with reference video conditioning where:
    - Reference latents (clean) are concatenated with target latents (noised)
    - Video coordinates are doubled to handle concatenated sequence
    - Loss is computed only on the target portion (masked loss)
    - Supports first frame conditioning on the target sequence
    """

    def __init__(self, conditioning_config: ConditioningConfig):
        """Initialize with configurable reference latents directory.

        Args:
            conditioning_config: Configuration for conditioning behavior
        """
        super().__init__(conditioning_config)

    def get_data_sources(self) -> dict[str, str]:
        """IC-LoRA training requires latents, conditions, and reference latents."""
        data_sources = {
            "latents": "latents",
            "conditions": "conditions",
            self.conditioning_config.reference_latents_dir: "ref_latents",
        }

        # Add artifact masks (noise map) if enabled
        if self.conditioning_config.generate_noisemap:
            if self.conditioning_config.artifact_masks_dir is None:
                raise ValueError("artifact_masks_dir must be specified when generate_noisemap is True")
            data_sources[self.conditioning_config.artifact_masks_dir] = "artifact_masks"

        return data_sources

    def prepare_batch(self, batch: dict[str, dict[str, Tensor]], timestep_sampler: TimestepSampler) -> TrainingBatch:
        """Prepare batch for IC-LoRA training with reference videos."""
        # Get pre-encoded latents
        latents = batch["latents"]
        target_latents = latents["latents"]
        ref_latents = batch["ref_latents"]["latents"]

        # Note: Batch sizes > 1 are partially supported, assuming
        # num_frames, height, width, fps are the same for all batch elements.
        latent_frames = latents["num_frames"][0].item()
        latent_height = latents["height"][0].item()
        latent_width = latents["width"][0].item()

        # Handle FPS with backward compatibility for old preprocessed datasets
        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values found in the batch. Found: {fps.tolist()}, using the first one: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        # Get pre-encoded text conditions
        conditions = batch["conditions"]
        prompt_embeds = conditions["prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        # Check if noise map generation is enabled
        generate_noisemap = self.conditioning_config.generate_noisemap
        if generate_noisemap:
            noisemap_latents = batch["artifact_masks"]["latents"]

        # Create noise for the target part (and noisemap if enabled)
        sigmas = timestep_sampler.sample_for(target_latents)
        noise_gt = torch.randn_like(target_latents, device=target_latents.device)
        sigmas = sigmas.view(-1, 1, 1)

        # Create conditioning mask
        batch_size = target_latents.shape[0]
        ref_seq_len = ref_latents.shape[1]
        target_seq_len = target_latents.shape[1]

        # Reference tokens are always conditioning
        ref_conditioning_mask = torch.ones(batch_size, ref_seq_len, dtype=torch.bool, device=target_latents.device)

        # Target tokens: check for first frame conditioning
        target_conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=target_seq_len,
            height=latent_height,
            width=latent_width,
            device=target_latents.device,
        )
        last_frame_mask = self._create_last_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=target_seq_len,
            height=latent_height,
            width=latent_width,
            device=target_latents.device,
        )
        target_conditioning_mask = target_conditioning_mask | last_frame_mask

        middle_frame_mask = self._create_middle_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=target_seq_len,
            num_frames=latent_frames,
            height=latent_height,
            width=latent_width,
            device=target_latents.device,
        )
        target_conditioning_mask = target_conditioning_mask | middle_frame_mask

        # Apply noise to target
        noisy_target = (1 - sigmas) * target_latents + sigmas * noise_gt

        # For first frame conditioning in target, use clean latents instead of noisy ones
        target_conditioning_mask_expanded = target_conditioning_mask.unsqueeze(-1)  # (B, target_seq_len, 1)
        noisy_target = torch.where(target_conditioning_mask_expanded, target_latents, noisy_target)

        # Prepare targets
        targets_gt = noise_gt - target_latents

        if generate_noisemap:
            # MODIFIED: Noisemap is now used as conditioning (clean, not noised)
            # Do NOT add noise to noisemap - keep it clean for spatial guidance
            # noisy_noisemap is NOT needed, use noisemap_latents directly

            # Apply feature scaling to reduce noisemap influence relative to reference
            noisemap_scale = self.conditioning_config.noisemap_feature_scale
            if noisemap_scale != 1.0:
                noisemap_latents = noisemap_latents * noisemap_scale
                logger.debug(f"Applied noisemap feature scale: {noisemap_scale}")

            # MODIFIED: Noisemap tokens are conditioning (frozen, no denoising needed)
            noisemap_conditioning_mask = torch.ones(batch_size, target_seq_len, dtype=torch.bool, device=target_latents.device)

            # Sequence order: reference + target + noisemap (same as before)
            # But now noisemap is clean conditioning instead of noisy target
            combined_latents = torch.cat([ref_latents, noisy_target, noisemap_latents], dim=1)

            # MODIFIED: Combine conditioning masks: ref (all 1) + target (mixed) + noisemap (all 1)
            conditioning_mask = torch.cat([ref_conditioning_mask, target_conditioning_mask, noisemap_conditioning_mask], dim=1)

            # MODIFIED: Targets only contain GT (noisemap is not a denoising target anymore)
            targets = targets_gt

            sequence_multiplier = 3  # reference + target + noisemap
        else:
            # Original behavior: reference + target only
            combined_latents = torch.cat([ref_latents, noisy_target], dim=1)
            conditioning_mask = torch.cat([ref_conditioning_mask, target_conditioning_mask], dim=1)
            targets = targets_gt
            sequence_multiplier = 2  # reference + target

        # Create timesteps based on conditioning mask
        sampled_timestep_values = torch.round(sigmas.squeeze(-1).squeeze(-1) * 1000.0).long()
        timesteps = self._create_timesteps_from_conditioning_mask(conditioning_mask, sampled_timestep_values)

        # Use existing utility function for ROPE scale factors
        rope_scale_factors = get_rope_scale_factors(fps)

        # Prepare video coordinates with appropriate sequence multiplier
        batch_size = combined_latents.shape[0]
        raw_video_coords = prepare_video_coordinates(
            num_frames=latent_frames,
            height=latent_height,
            width=latent_width,
            batch_size=batch_size,
            sequence_multiplier=sequence_multiplier,
            device=target_latents.device,
        )

        # Apply pre-scaling to raw coordinates.
        # The LTXVideoRotaryPosEmbed expects video_coords to be (B, 3, SeqLen) if provided.
        # It then divides video_coords[:, 0] by base_num_frames, etc.
        # So, the video_coords we pass should be: raw_coord * rope_interpolation_factor
        prescaled_f = raw_video_coords[..., 0] * rope_scale_factors[0]
        prescaled_h = raw_video_coords[..., 1] * rope_scale_factors[1]
        prescaled_w = raw_video_coords[..., 2] * rope_scale_factors[2]

        # Stack to (B, 3, seq_multiplier*F*H*W) for the transformer's video_coords argument
        video_coords = torch.stack([prescaled_f, prescaled_h, prescaled_w], dim=1)

        return TrainingBatch(
            latents=combined_latents,
            targets=targets,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            timesteps=timesteps,
            sigmas=sigmas,
            conditioning_mask=conditioning_mask,
            num_frames=latent_frames,
            height=latent_height,
            width=latent_width,
            fps=fps,
            rope_interpolation_scale=rope_scale_factors,
            video_coords=video_coords,
        )

    def compute_loss(self, model_pred: Tensor, batch: TrainingBatch) -> Tensor | dict[str, Tensor]:
        """Compute masked loss only on target portion, excluding conditioning tokens.

        Returns:
            If generate_noisemap is True: dict with keys 'total', 'gt' (for logging compatibility)
            Otherwise: Tensor (scalar loss)
        """
        if self.conditioning_config.generate_noisemap:
            # MODIFIED: With noise map as conditioning
            # Sequence: [ref, GT, noisemap]
            # batch.targets shape: [B, target_seq_len, C] (only GT, noisemap is not a target)
            # model_pred shape: [B, ref_seq_len + target_seq_len + noisemap_seq_len, C]

            # Calculate sequence lengths
            gt_seq_len = batch.targets.shape[1]  # Only GT
            noisemap_seq_len = gt_seq_len  # Noisemap has same length as GT

            # Get reference sequence length from conditioning mask
            # conditioning_mask: [ref(all 1), GT(mixed), noisemap(all 1)]
            ref_seq_len = batch.conditioning_mask.shape[1] - gt_seq_len - noisemap_seq_len

            # Extract only GT predictions from model output
            # Sequence positions: [0:ref_seq_len] = ref, [ref_seq_len:ref_seq_len+gt_seq_len] = GT
            gt_pred = model_pred[:, ref_seq_len:ref_seq_len + gt_seq_len]

            # GT target (batch.targets already only contains GT)
            gt_target = batch.targets

            # Extract GT conditioning mask
            gt_conditioning_mask = batch.conditioning_mask[:, ref_seq_len:ref_seq_len + gt_seq_len]

            # Compute GT loss (with conditioning mask applied)
            gt_loss = (gt_pred - gt_target).pow(2)
            gt_loss_mask = (~gt_conditioning_mask.unsqueeze(-1)).float()
            gt_loss = gt_loss.mul(gt_loss_mask).div(gt_loss_mask.mean()).mean()

            # Return only GT loss (noisemap is not trained)
            # Keep dict format for logging compatibility
            return {
                'total': gt_loss,
                'gt': gt_loss,
            }
        else:
            # Original behavior: single target without noise map
            target_seq_len = batch.targets.shape[1]
            target_pred = model_pred[:, -target_seq_len:]
            target_conditioning_mask = batch.conditioning_mask[:, -target_seq_len:]

            loss = (target_pred - batch.targets).pow(2)

            # Create loss mask: exclude conditioning tokens
            loss_mask = (~target_conditioning_mask.unsqueeze(-1)).float()

            # Apply original loss computation pattern
            loss = loss.mul(loss_mask).div(loss_mask.mean())
            return loss.mean()


def get_training_strategy(conditioning_config: ConditioningConfig) -> TrainingStrategy:
    """Factory function to create the appropriate training strategy.

    Args:
        conditioning_config: Configuration for conditioning behavior

    Returns:
        The appropriate training strategy instance

    Raises:
        ValueError: If conditioning mode is not supported
    """
    conditioning_mode = conditioning_config.mode

    if conditioning_mode == "none":
        strategy = StandardTrainingStrategy(conditioning_config)
    elif conditioning_mode == "reference_video":
        strategy = ReferenceVideoTrainingStrategy(conditioning_config)
    else:
        raise ValueError(f"Unknown conditioning mode: {conditioning_mode}")

    logger.debug(f"🎯 Using {strategy.__class__.__name__}")
    return strategy
