"""AutoModel implementation for loading pretrained models."""

import os
from typing import Dict, Tuple, Optional

import jax.numpy as jnp
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

from ..models import llama_model
from ..models.llama_config import Llama3_1_8B_Configurator, create_llama_model


MODEL_CONFIG = {
    "llama-3.1-8B-Instruct-JAX": {
        "hf_model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "felafax_model_name": "felafax/llama-3.1-8B-Instruct-JAX",
        "chkpt_filename": "llama-3.1-8B-Instruct-JAX.flax",
    }
}


class AutoJAXModelForCausalLM:
    """Auto class for loading pretrained JAX models."""

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        huggingface_token: Optional[str] = None,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        lora_rank: int = 8,
        lora_alpha: float = 16,
        **kwargs,
    ) -> Tuple[str, llama_model.CausalLlamaModule, Llama3_1_8B_Configurator, AutoTokenizer]:
        """Downloads model from HF and returns model path, model, config, and tokenizer.
        
        Args:
            model_name: Name of the model to load
            huggingface_token: Optional HF token for private models
            dtype: Data type for model computations
            param_dtype: Data type for model parameters
            lora_rank: Rank for LoRA adaptation
            lora_alpha: Alpha scaling for LoRA
            **kwargs: Additional args passed to model initialization
            
        Returns:
            Tuple containing:
            - Path to downloaded model checkpoint
            - Initialized model
            - Model configuration
            - Tokenizer
        """
        print(f"Downloading model {model_name}...")
        
        try:
            download_config = MODEL_CONFIG[model_name]
        except KeyError:
            raise ValueError(
                f"Invalid model name: {model_name}. "
                f"Available models are: {', '.join(MODEL_CONFIG.keys())}"
            )

        # Download model files
        model_dir = snapshot_download(
            repo_id=download_config["felafax_model_name"],
            token=huggingface_token,
        )
        model_path = os.path.join(model_dir, download_config["chkpt_filename"])

        # Load tokenizer
        if huggingface_token:
            tokenizer = AutoTokenizer.from_pretrained(
                download_config["hf_model_name"], 
                token=huggingface_token
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"{model_name} was downloaded to {model_path}.")

        # Initialize model configuration and model
        llama_model_configurator = create_llama_model(model_name)
        llama_model_config = llama_model_configurator.get_model_config()
        llama_model_hf_config = llama_model_configurator.get_hf_pretrained_config(
            llama_model_config)

        model = llama_model.CausalLlamaModule(
            llama_model_hf_config,
            dtype=dtype,
            param_dtype=param_dtype,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

        return model_path, model, llama_model_configurator, tokenizer
