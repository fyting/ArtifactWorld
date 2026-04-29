from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from ltxv_trainer.model_loader import LtxvModelVersion
from ltxv_trainer.quantization import QuantizationOptions


class ConfigBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(ConfigBaseModel):
    """Configuration for the base model and training mode"""

    model_source: str | Path | LtxvModelVersion = Field(
        default=LtxvModelVersion.latest(),
        description="Model source - can be a HuggingFace repo ID, local path, or LtxvModelVersion",
    )

    training_mode: Literal["lora", "full"] = Field(
        default="lora",
        description="Training mode - either LoRA fine-tuning or full model fine-tuning",
    )

    load_checkpoint: str | Path | None = Field(
        default=None,
        description="Path to a checkpoint file or directory to load from. "
        "If a directory is provided, the latest checkpoint will be used.",
    )

    scheduler_source: str | Path | None = Field(
        default=None,
        description="Optional explicit path/repo for scheduler component",
    )

    tokenizer_source: str | Path | None = Field(
        default=None,
        description="Optional explicit path/repo for tokenizer component",
    )

    text_encoder_source: str | Path | None = Field(
        default=None,
        description="Optional explicit path/repo for text encoder component",
    )

    vae_source: str | Path | LtxvModelVersion | None = Field(
        default=None,
        description="Optional explicit source for VAE component",
    )

    transformer_source: str | Path | LtxvModelVersion | None = Field(
        default=None,
        description="Optional explicit source for transformer component",
    )

    # noinspection PyNestedDecorators
    @field_validator("model_source", mode="before")
    @classmethod
    def validate_model_source(cls, v):  # noqa: ANN001, ANN206
        """Try to convert model source to LtxvModelVersion if possible."""
        if isinstance(v, (str, LtxvModelVersion)):
            try:
                return LtxvModelVersion(v)
            except ValueError:
                return v
        return v


class LoraConfig(ConfigBaseModel):
    """Configuration for LoRA fine-tuning"""

    rank: int = Field(
        default=64,
        description="Rank of LoRA adaptation",
        ge=2,
    )

    alpha: int = Field(
        default=64,
        description="Alpha scaling factor for LoRA",
        ge=1,
    )

    dropout: float = Field(
        default=0.0,
        description="Dropout probability for LoRA layers",
        ge=0.0,
        le=1.0,
    )

    target_modules: list[str] = Field(
        default=("to_k", "to_q", "to_v", "to_out.0"),
        description="List of modules to target with LoRA",
    )


class ConditioningConfig(ConfigBaseModel):
    """Configuration for conditioning during training"""

    mode: Literal["none", "reference_video"] = Field(
        default="none",
        description="Type of conditioning to use during training",
    )

    first_frame_conditioning_p: float = Field(
        default=0.1,
        description="Probability of conditioning on the first frame during training",
        ge=0.0,
        le=1.0,
    )

    last_frame_conditioning_p: float = Field(
        default=0.0,
        description="Probability of conditioning on the last frame during training",
        ge=0.0,
        le=1.0,
    )

    middle_frame_conditioning_p: float = Field(
        default=0.0,
        description="Probability of conditioning on a random middle frame during training",
        ge=0.0,
        le=1.0,
    )

    reference_latents_dir: str = Field(
        default="ref_latents",
        description="Directory name for latents of reference videos when using reference_video mode",
    )

    artifact_masks_dir: str | None = Field(
        default=None,
        description="Directory name for artifact mask latents (noise map). Required when generate_noisemap is True",
    )

    generate_noisemap: bool = Field(
        default=False,
        description="Enable noise map generation during training and inference. When True, model will output both cleaned video and noise map",
    )

    noisemap_loss_weight: float = Field(
        default=0.5,
        description="Weight for noise map loss in total loss computation",
        ge=0.0,
        le=10.0,
    )

    noisemap_feature_scale: float = Field(
        default=1.0,
        description="Feature scale for noisemap latents. Values < 1.0 reduce noisemap influence relative to reference video. Must be consistent between training and inference.",
        ge=0.0,
    )


class OptimizationConfig(ConfigBaseModel):
    """Configuration for optimization parameters"""

    learning_rate: float = Field(
        default=5e-4,
        description="Learning rate for optimization",
    )

    steps: int = Field(
        default=3000,
        description="Number of training steps",
    )

    batch_size: int = Field(
        default=2,
        description="Batch size for training",
    )

    gradient_accumulation_steps: int = Field(
        default=1,
        description="Number of steps to accumulate gradients",
    )

    max_grad_norm: float = Field(
        default=1.0,
        description="Maximum gradient norm for clipping",
    )

    optimizer_type: Literal["adamw", "adamw8bit"] = Field(
        default="adamw",
        description="Type of optimizer to use for training",
    )

    scheduler_type: Literal[
        "constant",
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
    ] = Field(
        default="linear",
        description="Type of scheduler to use for training",
    )

    scheduler_params: dict = Field(
        default_factory=dict,
        description="Parameters for the scheduler",
    )

    enable_gradient_checkpointing: bool = Field(
        default=False,
        description="Enable gradient checkpointing to save memory at the cost of slower training",
    )


class AccelerationConfig(ConfigBaseModel):
    """Configuration for hardware acceleration and compute optimization"""

    mixed_precision_mode: Literal["no", "fp16", "bf16"] | None = Field(
        default="bf16",
        description="Mixed precision training mode",
    )

    quantization: QuantizationOptions | None = Field(
        default=None,
        description="Quantization precision to use",
    )

    load_text_encoder_in_8bit: bool = Field(
        default=False,
        description="Whether to load the text encoder in 8-bit precision to save memory",
    )

    compile_with_inductor: bool = Field(
        default=True,
        description="Compile the model with Torch Inductor",
    )

    compilation_mode: Literal["default", "reduce-overhead", "max-autotune"] = Field(
        default="reduce-overhead",
        description="Compilation mode for Torch Inductor",
    )


class DataConfig(ConfigBaseModel):
    """Configuration for data loading and processing"""

    preprocessed_data_root: str = Field(
        description="Path to folder containing preprocessed training data",
    )

    num_dataloader_workers: int = Field(
        default=2,
        description="Number of background processes for data loading (0 means synchronous loading)",
        ge=0,
    )


class ValidationConfig(ConfigBaseModel):
    """Configuration for validation during training"""

    prompts: list[str] = Field(
        default_factory=list,
        description="List of prompts to use for validation",
    )

    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        description="Negative prompt to use for validation examples",
    )

    images: list[str] | None = Field(
        default=None,
        description="List of image paths to use for validation. "
        "One image path must be provided for each validation prompt",
    )

    images_end: list[str] | None = Field(
        default=None,
        description="List of end-frame image paths to use for validation. "
        "One image path must be provided for each validation prompt",
    )

    reference_videos: list[str] | None = Field(
        default=None,
        description="List of reference video paths to use for validation. "
        "One video path must be provided for each validation prompt",
    )

    noisemap_videos: list[str] | None = Field(
        default=None,
        description="List of noisemap video paths to use for validation when generate_noisemap is enabled. "
        "One video path must be provided for each validation prompt",
    )

    auxiliary_latent_path: str | None = Field(
        default=None,
        description="Path to auxiliary video latent file A (.pt) for blending with noisemap_videos during inference. "
        "E.g., /path/to/seg_0020_white.pt",
    )

    auxiliary_latent_b_path: str | None = Field(
        default=None,
        description="Path to auxiliary video latent file B (.pt) for blending with latent A and noisemap_videos during inference. "
        "E.g., /path/to/seg_0020_black.pt",
    )

    noisemap_blend_weights: list[tuple[float, float, float]] | None = Field(
        default=None,
        description="List of (auxiliary_a_weight, auxiliary_b_weight, noisemap_weight) tuples for each inference step. "
        "Each tuple specifies the blending weights where auxiliary_a_weight + auxiliary_b_weight + noisemap_weight = 1.0. "
        "The length of this list should match inference_steps. "
        "If auxiliary_latent_b_path is not provided, use 2-tuple format (auxiliary_weight, noisemap_weight). "
        "If None, only noisemap_videos will be used without blending.",
    )

    save_noisemap_latents: bool = Field(
        default=False,
        description="If True, save the encoded noisemap latents to a .pt file in the output directory. "
        "Useful for debugging or reusing the same latents later.",
    )

    video_dims: tuple[int, int, int] = Field(
        default=(704, 480, 161),
        description="Dimensions of validation videos (width, height, frames)",
    )

    seed: int = Field(
        default=42,
        description="Random seed used when sampling validation videos",
    )

    inference_steps: int = Field(
        default=50,
        description="Number of inference steps for validation",
        gt=0,
    )

    interval: int | None = Field(
        default=100,
        description="Number of steps between validation runs. If None, validation is disabled.",
        gt=0,
    )

    videos_per_prompt: int = Field(
        default=1,
        description="Number of videos to generate per validation prompt",
        gt=0,
    )

    guidance_scale: float = Field(
        default=3.5,
        description="Guidance scale to use during validation",
        ge=1.0,
    )

    skip_initial_validation: bool = Field(
        default=False,
        description="Skip validation video sampling at step 0 (beginning of training)",
    )

    remove_last_n_frames: int | None = Field(
        default=None,
        description="Number of frames to remove from the end of the generated video before saving",
        ge=0,
    )

    @field_validator("images")
    @classmethod
    def validate_num_images(cls, v: list[str] | None, info: ValidationInfo) -> list[str] | None:
        """Validate that number of images (if provided) matches number of prompts."""
        num_prompts = len(info.data.get("prompts", []))
        if v is not None and len(v) != num_prompts:
            raise ValueError(f"Number of images ({len(v)}) must match number of prompts ({num_prompts})")
        return v

    @field_validator("images_end")
    @classmethod
    def validate_num_images_end(cls, v: list[str] | None, info: ValidationInfo) -> list[str] | None:
        """Validate that number of images_end (if provided) matches number of prompts."""
        num_prompts = len(info.data.get("prompts", []))
        if v is not None and len(v) != num_prompts:
            raise ValueError(f"Number of images_end ({len(v)}) must match number of prompts ({num_prompts})")
        return v

    @field_validator("reference_videos")
    @classmethod
    def validate_num_reference_videos(cls, v: list[str] | None, info: ValidationInfo) -> list[str] | None:
        """Validate that number of reference videos (if provided) matches number of prompts."""
        num_prompts = len(info.data.get("prompts", []))
        if v is not None and len(v) != num_prompts:
            raise ValueError(f"Number of reference videos ({len(v)}) must match number of prompts ({num_prompts})")
        return v

    @model_validator(mode="after")
    def validate_blend_weights(self) -> "ValidationConfig":
        """Validate that blend weights sum to 1.0 and match inference_steps."""
        if self.noisemap_blend_weights is None:
            return self

        # Validate length matches inference_steps
        if len(self.noisemap_blend_weights) != self.inference_steps:
            raise ValueError(
                f"Number of blend weight tuples ({len(self.noisemap_blend_weights)}) "
                f"must match inference_steps ({self.inference_steps})"
            )

        # Check if using 3-tuple (with latent B) or 2-tuple (without latent B)
        has_latent_b = self.auxiliary_latent_b_path is not None
        expected_tuple_len = 3 if has_latent_b else 2

        # Validate each weight tuple
        for i, weights in enumerate(self.noisemap_blend_weights):
            if len(weights) != expected_tuple_len:
                raise ValueError(
                    f"Expected {expected_tuple_len}-tuple at step {i} "
                    f"(auxiliary_latent_b_path is {'set' if has_latent_b else 'not set'}), "
                    f"but got {len(weights)}-tuple: {weights}"
                )

            # Check all weights are non-negative
            if any(w < 0 for w in weights):
                raise ValueError(f"Weights at step {i} must be non-negative, got {weights}")

            # Check sum is 1.0
            weight_sum = sum(weights)
            if abs(weight_sum - 1.0) > 1e-5:  # Allow small floating point errors
                raise ValueError(f"Weights at step {i} must sum to 1.0, got {weight_sum} from {weights}")

        return self


class CheckpointsConfig(ConfigBaseModel):
    """Configuration for model checkpointing during training"""

    interval: int | None = Field(
        default=None,
        description="Number of steps between checkpoint saves. If None, intermediate checkpoints are disabled.",
        gt=0,
    )

    keep_last_n: int = Field(
        default=1,
        description="Number of most recent checkpoints to keep. Set to -1 to keep all checkpoints.",
        ge=-1,
    )


class HubConfig(ConfigBaseModel):
    """Configuration for Hugging Face Hub integration"""

    push_to_hub: bool = Field(default=False, description="Whether to push the model weights to the Hugging Face Hub")
    hub_model_id: str | None = Field(
        default=None, description="Hugging Face Hub repository ID (e.g., 'username/repo-name')"
    )

    @model_validator(mode="after")
    def validate_hub_config(self) -> "HubConfig":
        """Validate that hub_model_id is not None when push_to_hub is True."""
        if self.push_to_hub and not self.hub_model_id:
            raise ValueError("hub_model_id must be specified when push_to_hub is True")
        return self


class WandbConfig(ConfigBaseModel):
    """Configuration for Weights & Biases logging"""

    enabled: bool = Field(
        default=False,
        description="Whether to enable W&B logging",
    )

    project: str = Field(
        default="ltxv-trainer",
        description="W&B project name",
    )

    entity: str | None = Field(
        default=None,
        description="W&B username or team",
    )

    tags: list[str] = Field(
        default_factory=list,
        description="Tags to add to the W&B run",
    )

    log_validation_videos: bool = Field(
        default=True,
        description="Whether to log validation videos to W&B",
    )


class FlowMatchingConfig(ConfigBaseModel):
    """Configuration for flow matching training"""

    timestep_sampling_mode: Literal["uniform", "shifted_logit_normal"] = Field(
        default="shifted_logit_normal",
        description="Mode to use for timestep sampling",
    )

    timestep_sampling_params: dict = Field(
        default_factory=dict,
        description="Parameters for timestep sampling",
    )


class LtxvTrainerConfig(ConfigBaseModel):
    """Unified configuration for LTXV training"""

    # Sub-configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoraConfig | None = Field(default=None)
    conditioning: ConditioningConfig = Field(default_factory=ConditioningConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    acceleration: AccelerationConfig = Field(default_factory=AccelerationConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    checkpoints: CheckpointsConfig = Field(default_factory=CheckpointsConfig)
    hub: HubConfig = Field(default_factory=HubConfig)
    flow_matching: FlowMatchingConfig = Field(default_factory=FlowMatchingConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    # General configuration
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )

    output_dir: str = Field(
        default="outputs",
        description="Directory to save model outputs",
    )

    # noinspection PyNestedDecorators
    @field_validator("output_dir")
    @classmethod
    def expand_output_path(cls, v: str) -> str:
        """Expand user home directory in output path."""
        return str(Path(v).expanduser().resolve())

    @model_validator(mode="after")
    def validate_conditioning_compatibility(self) -> "LtxvTrainerConfig":
        """Validate that conditioning and validation configurations are compatible."""

        # Check that reference videos are provided when using reference_video conditioning
        if self.conditioning.mode == "reference_video" and self.validation.reference_videos is None:
            raise ValueError(
                "reference_videos must be provided in validation config when conditioning.mode is 'reference_video'"
            )

        # Check that LoRA config is provided when training mode is lora
        if self.model.training_mode == "lora" and self.lora is None:
            raise ValueError("LoRA configuration must be provided when training_mode is 'lora'")

        # Check that LoRA config is provided when using reference_video conditioning with LoRA training mode
        if self.conditioning.mode == "reference_video" and self.model.training_mode != "lora":
            raise ValueError("Training mode must be 'lora' when using reference_video conditioning")

        return self
