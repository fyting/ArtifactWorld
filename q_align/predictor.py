"""Batch video-artifact predictor.

Encodes each video once, then scores every artifact prompt in one generate call.
On CUDA OOM it falls back to one prompt at a time with the same cached features.
"""

import os

import torch
import torch.nn.functional as F
from PIL import Image
from transformers.modeling_outputs import CausalLMOutputWithPast

from q_align import conversation as conversation_lib
from q_align.artifacts import ARTIFACTS
from q_align.constants import IMAGE_TOKEN_INDEX
from q_align.load_video import load_video
from q_align.mm_utils import tokenizer_image_token
from q_align.model.builder import load_pretrained_model
from q_align.model.modeling_mplug_owl2 import MPLUGOwl2LlamaForCausalLM


def _prepare_inputs_for_generation_override(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values:
        input_ids = input_ids[:, -1:]

    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "images": kwargs.get("images", None),
            "modality_indicators": kwargs.get("modality_indicators", None),
        }
    )
    return model_inputs


_original_forward = MPLUGOwl2LlamaForCausalLM.forward


def _forward_override(
    self,
    input_ids=None,
    attention_mask=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    images=None,
    return_dict=None,
    modality_indicators=None,
):
    if inputs_embeds is not None and modality_indicators is not None and images is None:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=None,
            modality_indicators=modality_indicators,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        if not return_dict:
            return (logits,) + outputs[1:]
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    return _original_forward(
        self,
        input_ids,
        attention_mask,
        past_key_values,
        inputs_embeds,
        labels,
        use_cache,
        output_attentions,
        output_hidden_states,
        images,
        return_dict,
    )


MPLUGOwl2LlamaForCausalLM.prepare_inputs_for_generation = _prepare_inputs_for_generation_override
MPLUGOwl2LlamaForCausalLM.forward = _forward_override


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    result = Image.new(pil_img.mode, (height, height), background_color)
    result.paste(pil_img, ((height - width) // 2, 0))
    return result


def default_prompt(artifact_name):
    """Prompt used when labeling all artifacts. Matches the A100 script default."""
    return f"USER: <|image|>Does this video suffer from {artifact_name}?\nASSISTANT:"


def subset_prompt(artifact_name):
    """Prompt used when --artifacts is set. Matches the A100 worker."""
    question = f"<|image|>Does this video suffer from {artifact_name}?"
    if not question.endswith("?"):
        question = question + "?"
    question = question + " Please answer yes or no."
    conv = conversation_lib.default_conversation.copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


class ArtifactPredictor:
    def __init__(
        self,
        model_path,
        model_base,
        device="cuda:0",
        use_8bit=False,
        max_frames=60,
    ):
        self.max_frames = max_frames
        self.device = device

        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Artifact checkpoint not found: {model_path}")
        if not os.path.isdir(model_base):
            raise FileNotFoundError(
                f"Base model not found: {model_base}. "
                "Expected the vendored one-align directory."
            )
        adapter_ok = any(
            os.path.exists(os.path.join(model_path, name))
            for name in ("adapter_model.bin", "adapter_model.safetensors", "adapter_config.json")
        )
        if not adapter_ok:
            raise FileNotFoundError(f"No LoRA adapter files in {model_path}")
        if not os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
            raise FileNotFoundError(f"Missing non_lora_trainables.bin in {model_path}")

        conversation_lib.default_conversation = conversation_lib.conv_templates["v1"]

        device_map = {"": device}
        print(f"Loading base model from {model_base}")
        print(f"Loading artifact LoRA from {model_path} onto {device}")
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path,
            model_base,
            "q-align-lora",
            device=device,
            device_map=device_map,
            load_8bit=use_8bit,
        )
        self.tokenizer.padding_side = "left"
        self.model.eval()

        yes_tokens = self.tokenizer(["Yes", "No"])["input_ids"]
        self.yes_id = yes_tokens[0][-1]
        self.no_id = yes_tokens[1][-1]

    def _model_device(self):
        if hasattr(self.model, "device"):
            return self.model.device
        if getattr(self.model, "hf_device_map", None):
            return list(self.model.hf_device_map.values())[0]
        return torch.device(self.device)

    def _load_frames(self, video_path):
        images = load_video(video_path)
        if self.max_frames is not None and len(images) > self.max_frames:
            images = images[: self.max_frames]
        background = tuple(int(x * 255) for x in self.image_processor.image_mean)
        images = [expand2square(img, background) for img in images]
        return self.image_processor.preprocess(images, return_tensors="pt")["pixel_values"].half()

    def _prompts_for(self, artifact_names):
        if artifact_names is None:
            names = list(ARTIFACTS)
            return names, [default_prompt(name) for name in names]
        unknown = [name for name in artifact_names if name not in ARTIFACTS]
        if unknown:
            raise ValueError(f"Unknown artifacts: {unknown}. Valid: {ARTIFACTS}")
        return list(artifact_names), [subset_prompt(name) for name in artifact_names]

    def _score_generation(self, generated_ids, first_token_logits, prompt):
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip().lower()
        yes_score = first_token_logits[self.yes_id].item()
        no_score = first_token_logits[self.no_id].item()
        probs = F.softmax(torch.tensor([yes_score, no_score]), dim=0)
        if "yes" in generated_text:
            pred = "Yes"
        elif "no" in generated_text:
            pred = "No"
        else:
            pred = "Yes" if yes_score > no_score else "No"
        return {
            "prediction": pred,
            "prob_yes": probs[0].item(),
            "prob_no": probs[1].item(),
            "prompt": prompt,
        }

    def _build_embeds(self, input_ids, video_features_flattened, model_device):
        image_indices = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0]
        if len(image_indices) == 0:
            embeds = self.model.model.embed_tokens(input_ids)
            modality = torch.zeros(embeds.shape[0], dtype=torch.long, device=model_device)
            return embeds, modality

        image_start = image_indices[0]
        part1 = self.model.model.embed_tokens(input_ids[:image_start])
        part2 = self.model.model.embed_tokens(input_ids[image_start + 1 :])
        embeds = torch.cat([part1, video_features_flattened, part2], dim=0)
        modality = torch.cat(
            [
                torch.zeros(part1.shape[0], dtype=torch.long, device=model_device),
                torch.ones(video_features_flattened.shape[0], dtype=torch.long, device=model_device),
                torch.zeros(part2.shape[0], dtype=torch.long, device=model_device),
            ],
            dim=0,
        )
        return embeds, modality

    def predict(self, video_path, artifacts=None):
        try:
            image_tensor = self._load_frames(video_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to read video {video_path}: {exc}") from exc

        artifact_names, prompts = self._prompts_for(artifacts)
        model_device = self._model_device()
        image_tensor = image_tensor.to(model_device)

        try:
            with torch.inference_mode():
                video_features = self.model.encode_images(image_tensor)
                video_features_flattened = video_features.flatten(0, 1).to(model_device)

            batch_embeds = []
            batch_modality = []
            for prompt in prompts:
                ids = tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                ).to(model_device)
                embeds, modality = self._build_embeds(ids, video_features_flattened, model_device)
                batch_embeds.append(embeds)
                batch_modality.append(modality)

            max_len = max(item.shape[0] for item in batch_embeds)
            padded_embeds = []
            padded_modality = []
            padded_mask = []
            for embeds, modality in zip(batch_embeds, batch_modality):
                pad_len = max_len - embeds.shape[0]
                if pad_len > 0:
                    pad_embeds = torch.zeros(
                        (pad_len, embeds.shape[1]), dtype=embeds.dtype, device=model_device
                    )
                    pad_modality = torch.zeros((pad_len,), dtype=modality.dtype, device=model_device)
                    pad_mask = torch.zeros((pad_len,), dtype=torch.long, device=model_device)
                    ones = torch.ones((embeds.shape[0],), dtype=torch.long, device=model_device)
                    padded_embeds.append(torch.cat([pad_embeds, embeds], dim=0))
                    padded_modality.append(torch.cat([pad_modality, modality], dim=0))
                    padded_mask.append(torch.cat([pad_mask, ones], dim=0))
                else:
                    padded_embeds.append(embeds)
                    padded_modality.append(modality)
                    padded_mask.append(
                        torch.ones((embeds.shape[0],), dtype=torch.long, device=model_device)
                    )

            inputs_embeds = torch.stack(padded_embeds)
            modality_indicators = torch.stack(padded_modality)
            attention_mask = torch.stack(padded_mask)

            with torch.inference_mode():
                outputs = self.model.generate(
                    inputs_embeds=inputs_embeds,
                    modality_indicators=modality_indicators,
                    attention_mask=attention_mask,
                    max_new_tokens=5,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    use_cache=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                    images=None,
                )

            input_len = inputs_embeds.shape[1]
            results = {}
            for i, name in enumerate(artifact_names):
                generated_ids = outputs.sequences[i, input_len:]
                results[name] = self._score_generation(
                    generated_ids, outputs.scores[0][i], prompts[i]
                )
            return results
        except torch.cuda.OutOfMemoryError:
            print(f"OOM on batch inference for {video_path}. Falling back to sequential prompts.")
            torch.cuda.empty_cache()
            return self._predict_sequential(image_tensor, artifact_names, prompts, model_device)

    def _predict_sequential(self, image_tensor, artifact_names, prompts, model_device):
        image_tensor = image_tensor.to(model_device, dtype=torch.float16)
        with torch.inference_mode():
            video_features = self.model.encode_images(image_tensor)
            video_features_flattened = video_features.flatten(0, 1).to(model_device)

        results = {}
        for name, prompt in zip(artifact_names, prompts):
            input_ids = tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).to(model_device)
            embeds, modality = self._build_embeds(input_ids, video_features_flattened, model_device)
            inputs_embeds = embeds.unsqueeze(0)
            modality_indicators = modality.unsqueeze(0)
            attention_mask = torch.ones(
                (1, embeds.shape[0]), dtype=torch.long, device=model_device
            )
            with torch.inference_mode():
                outputs = self.model.generate(
                    inputs_embeds=inputs_embeds,
                    modality_indicators=modality_indicators,
                    attention_mask=attention_mask,
                    images=None,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    max_new_tokens=5,
                    use_cache=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            input_len = inputs_embeds.shape[1]
            generated_ids = outputs.sequences[0, input_len:]
            results[name] = self._score_generation(generated_ids, outputs.scores[0][0], prompt)
        return results
