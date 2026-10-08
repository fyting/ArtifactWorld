#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from transformers.models.clip.image_processing_clip import CLIPImageProcessor
import torch
from q_align.model import *
from icecream import ic
def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):
    kwargs = {"device_map": device_map}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
    if 'q-align' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')

        # Check if we are loading LoRA or just base model
        is_lora = 'lora' in model_name.lower() and model_path is not None and model_path != ""

        if is_lora and model_base is not None:
            # lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            # Use model_base config instead of model_path config if config.json is missing in LoRA
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_base, trust_remote_code=True, local_files_only=True)
            lora_cfg_pretrained._name_or_path = os.path.abspath(model_base)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, local_files_only=True)
            print('Loading mPLUG-Owl2 from base model...')
            model = MPLUGOwl2LlamaForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, local_files_only=True, **kwargs
            )
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional mPLUG-Owl2 weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
                print(f"Loaded non_lora_trainables with {len(non_lora_trainables)} keys")
                # print(list(non_lora_trainables.keys())[:5]) # Debug print
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')

            # Key matching logic: handle prefix mismatch
            # The saved keys might have 'base_model.model.model.' or just 'model.' prefix
            # We need to strip them to match 'model.' or '' depending on where they go

            # 1. Strip 'base_model.model.' prefix if present (from PEFT wrapping)
            non_lora_trainables = {(k[16:] if k.startswith('base_model.model.') else k): v for k, v in non_lora_trainables.items()}

            # 2. Also handle 'base_model.' just in case
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}

            # 2.5. Handle '.model.' prefix (some checkpoints may have this format)
            non_lora_trainables = {(k[7:] if k.startswith('.model.') else k): v for k, v in non_lora_trainables.items()}

            # 2.6. Add 'model.' prefix if keys start with 'visual_abstractor' but not 'model.visual_abstractor'
            # The model's state_dict expects 'model.visual_abstractor.*' format (see convert_mplug_owl2_weight_to_hf.py)
            non_lora_trainables = {('model.' + k if k.startswith('visual_abstractor') and not k.startswith('model.visual_abstractor') else k): v for k, v in non_lora_trainables.items()}

            # 3. If keys start with 'model.model.', strip one 'model.' (legacy Q-Align issue)
            # Check if we need to do this
            # if any(k.startswith('model.model.') for k in non_lora_trainables):
            #    non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}

            # Let's be more robust:
            # The model variable is MPLUGOwl2LlamaForCausalLM.
            # Its state_dict keys usually start with 'model.vision_model...' or 'model.layers...'
            # Let's inspect what we have in non_lora_trainables vs model.state_dict()

            # Attempt to load with relaxed strictness
            msg = model.load_state_dict(non_lora_trainables, strict=False)
            print(f"Loading non-LoRA weights result: {msg}")

            # Verify if visual abstractor weights changed?
            # Maybe not easy to check here, but msg.missing_keys shouldn't contain visual abstractor keys if loaded correctly.

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only OR just loading base model for inference
            print('Loading mPLUG-Owl2 from base model...')
            # cfg_pretrained = AutoConfig.from_pretrained(model_path)
            # Fallback to model_base config if model_path doesn't have it
            try:
                # If model_path is provided, try to load config from it
                if model_path:
                    cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                else:
                    raise OSError("model_path is empty")
            except OSError:
                cfg_pretrained = AutoConfig.from_pretrained(model_base, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, trust_remote_code=True)
            model = MPLUGOwl2LlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, trust_remote_code=True, **kwargs)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
            model = MPLUGOwl2LlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
    else:
        # Load language model
        if model_base is not None and model_path is not None and model_path != "":
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            # If model_path is None, use model_base if available
            path_to_load = model_path if (model_path is not None and model_path != "") else model_base
            tokenizer = AutoTokenizer.from_pretrained(path_to_load, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(path_to_load, low_cpu_mem_usage=True, **kwargs)

    #vision_tower = model.get_model().vision_model
    #print(vision_tower.device)
    #vision_tower.to(device=device, dtype=torch.float16)
    try:
        path_to_load = model_path if (model_path is not None and model_path != "") else model_base
        image_processor = CLIPImageProcessor.from_pretrained(path_to_load)
    except OSError:
        print(f"Failed to load image processor from {model_path}. Trying base model {model_base}...")
        if model_base is not None:
             image_processor = CLIPImageProcessor.from_pretrained(model_base)
        else:
             raise

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len