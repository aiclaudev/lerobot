#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import pad_vector
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

# New batch keys for hierarchical reasoning (training only). Hard-coded strings
# to avoid touching upstream lerobot constants.
OBS_LANGUAGE_AR_MASK = "observation.language.ar_mask"
OBS_LANGUAGE_LOSS_MASK = "observation.language.loss_mask"


@ProcessorStepRegistry.register(name="pi05_prepare_state_tokenizer_processor_step")
@dataclass
class Pi05PrepareStateTokenizerProcessorStep(ProcessorStep):
    """
    Processor step to prepare the state and tokenize the language input.
    """

    max_state_dim: int = 32
    task_key: str = "task"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05")
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        # TODO: check if this necessary
        state = deepcopy(state)

        # Prepare state (pad to max_state_dim)
        state = pad_vector(state, self.max_state_dim)

        # State should already be normalized to [-1, 1] by the NormalizerProcessorStep that runs before this step
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[i]))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            full_prompts.append(full_prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        # Normalize state to [-1, 1] range if needed (assuming it's already normalized by normalizer processor step!!)
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step does not alter the feature definitions.
        """
        return features


@ProcessorStepRegistry.register(name="pi05_reasoning_prepare_and_tokenize_step")
@dataclass
class Pi05ReasoningPrepareAndTokenizeStep(ProcessorStep):
    """Prepare state + tokenize a `Task: HL. Subtask: SUB. State: ...;\nAction: ` prompt.

    Subsumes Pi05PrepareStateTokenizerProcessorStep + TokenizerProcessorStep for
    the hierarchical-reasoning training path. Emits four batch tensors:

      - OBS_LANGUAGE_TOKENS         : int64[B, max_length]  full prompt token ids
      - OBS_LANGUAGE_ATTENTION_MASK : bool[B, max_length]   real vs padding
      - OBS_LANGUAGE_AR_MASK        : bool[B, max_length]   per-token causal block start
                                                            (1 at each subtask token + first tail token)
      - OBS_LANGUAGE_LOSS_MASK      : bool[B, max_length]   per-token CE loss target
                                                            (1 only at subtask tokens)

    When the batch has no `reasoning` (e.g. inference before AR decode),
    falls back to a subtask-empty prompt: `Task: HL. Subtask: <pad...>State: ...`
    and zero ar/loss masks — modeling code is responsible for two-stage handling.
    """

    tokenizer_name: str = "google/paligemma-3b-pt-224"
    max_length: int = 200
    max_state_dim: int = 32
    task_key: str = "task"
    reasoning_key: str = "reasoning"
    # Lazy-loaded tokenizer (not part of dataclass equality).
    _tokenizer: Any = field(default=None, init=False, repr=False, compare=False)

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
        return self._tokenizer

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05")
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")
        reasonings = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.reasoning_key)

        state = deepcopy(state)
        state = pad_vector(state, self.max_state_dim)
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        tok = self._get_tokenizer()
        bos_id = tok.bos_token_id
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

        B = len(tasks)
        all_tokens = np.full((B, self.max_length), pad_id, dtype=np.int64)
        all_attn = np.zeros((B, self.max_length), dtype=bool)
        all_ar = np.zeros((B, self.max_length), dtype=bool)
        all_loss = np.zeros((B, self.max_length), dtype=bool)

        for i, task in enumerate(tasks):
            hl = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[i]))
            sub = reasonings[i].strip() if (reasonings is not None and reasonings[i]) else ""

            prefix_a_str = f"Task: {hl}. Subtask: "
            tail_str = f". State: {state_str};\nAction: "

            a_ids = tok(prefix_a_str, add_special_tokens=False)["input_ids"]
            tail_ids = tok(tail_str, add_special_tokens=False)["input_ids"]
            sub_ids = tok(sub, add_special_tokens=False)["input_ids"] if sub else []

            # Compose: [BOS, prefix_a, subtask, tail]; truncate tail first if overflow.
            n_real = 1 + len(a_ids) + len(sub_ids) + len(tail_ids)
            if n_real > self.max_length:
                excess = n_real - self.max_length
                if excess <= len(tail_ids):
                    tail_ids = tail_ids[: len(tail_ids) - excess]
                else:
                    # Pathological — drop tail entirely, then subtask end. Keeps prompt parseable.
                    excess -= len(tail_ids)
                    tail_ids = []
                    sub_ids = sub_ids[: max(0, len(sub_ids) - excess)]
                n_real = 1 + len(a_ids) + len(sub_ids) + len(tail_ids)

            seq = [bos_id, *a_ids, *sub_ids, *tail_ids]
            all_tokens[i, :n_real] = seq
            all_attn[i, :n_real] = True

            sub_start = 1 + len(a_ids)
            sub_end = sub_start + len(sub_ids)
            # AR mask: each subtask token is its own causal block; first tail token opens its own block.
            all_ar[i, sub_start:sub_end] = True
            if len(tail_ids) > 0:
                all_ar[i, sub_end] = True
            # Loss mask: supervise only subtask tokens.
            all_loss[i, sub_start:sub_end] = True

        obs = transition.get(TransitionKey.OBSERVATION, {}).copy()
        obs[OBS_LANGUAGE_TOKENS] = torch.from_numpy(all_tokens)
        obs[OBS_LANGUAGE_ATTENTION_MASK] = torch.from_numpy(all_attn)
        obs[OBS_LANGUAGE_AR_MASK] = torch.from_numpy(all_ar)
        obs[OBS_LANGUAGE_LOSS_MASK] = torch.from_numpy(all_loss)
        transition[TransitionKey.OBSERVATION] = obs
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_pi05_pre_post_processors(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the PI0 policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Appending a newline character to the task description for tokenizer compatibility.
    5. Tokenizing the text prompt using the PaliGemma tokenizer.
    6. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the PI0 policy.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    # Add remaining processors
    base_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        AddBatchDimensionProcessorStep(),
        # NOTE: NormalizerProcessorStep MUST come before any Pi05*PrepareStateTokenizer* step
        # because those expect normalized state in [-1, 1] range for discretization
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]
    if getattr(config, "predict_reasoning", False):
        # Hierarchical reasoning: subsume the prepare+tokenize pair with a single step
        # that emits AR / loss masks alongside the token ids.
        input_steps: list[ProcessorStep] = [
            *base_steps,
            Pi05ReasoningPrepareAndTokenizeStep(
                tokenizer_name="google/paligemma-3b-pt-224",
                max_length=config.tokenizer_max_length,
                max_state_dim=config.max_state_dim,
            ),
            DeviceProcessorStep(device=config.device),
        ]
    else:
        input_steps = [
            *base_steps,
            Pi05PrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
            TokenizerProcessorStep(
                tokenizer_name="google/paligemma-3b-pt-224",
                max_length=config.tokenizer_max_length,
                padding_side="right",
                padding="max_length",
            ),
            DeviceProcessorStep(device=config.device),
        ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
