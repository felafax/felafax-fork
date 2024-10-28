"""Integration tests for the CausalLMTrainer.

Run with: pytest --capture=no tests/trainer/test_causal_lm_trainer.py
"""

import pytest
import jax.numpy as jnp
import optax
from dataclasses import dataclass

from felafax.trainer_engine.models.auto_model import AutoJAXModelForCausalLM
from felafax.trainer_engine.models.auto_model_config import AutoJAXModelConfig
from felafax.trainer_engine.trainer.causal_lm_trainer import trainer as trainer_lib
from felafax.trainer_engine.data.alpaca import AlpacaDataset
from felafax.trainer_engine import jax_utils


def test_trainer_single_step():
    """Tests that the trainer can successfully complete one training step."""
    model_name = "llama-3.1-8B-Instruct-JAX"

    # Obtain model configuration
    # model_config contains the model architecture params and partition rules.
    model_config = AutoJAXModelConfig.from_pretrained(model_name)
    
    # this should download the model weights and load it
    # model should be the equinox model 
    # this should use partitioning rules from the model_config
    model_with_params= AutoJAXModel.from_pretrained(
        model_name,
        model_config,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        lora_rank=8,
        lora_alpha=16,
        load_in_8bit=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name)
    trainer_config = TestTrainerConfig()
    
     # Setup dataset
    data_module = AlpacaDataset(
        batch_size=trainer_config.batch_size,
        max_seq_length=trainer_config.seq_length,
        max_examples=trainer_config.dataset_size_limit,
    )
    data_module.setup(tokenizer=tokenizer)

    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    trainer = SFTTrainer(
        model=model_with_params,
        model_config=model_config,
        tokenizer=tokenizer,
        train_dataset=train_dataloader,
        val_dataset=val_dataloader,
        dataset_text_field="text",
        max_seq_length=128,
        trainer_args=trainer_config,
    )

    # Run one training step
    try:
        state = trainer.train(
            train_dataloader,
            val_dataloader,
            run_jitted=True,
            platform="tpu",  # Adjust platform as needed
        )

        # Basic assertions to verify training occurred
        assert state is not None, "Training state should not be None"
        assert state.step == 1, "Should have completed exactly one step"

        # Check if parameters were updated
        assert state.lora_params is not None, "LoRA parameters should exist"
        assert state.opt_state is not None, "Optimizer state should exist"

    except Exception as e:
        pytest.fail(f"Training failed with error: {str(e)}")
