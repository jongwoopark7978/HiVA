#!/usr/bin/env python

import json
from collections import deque
from types import SimpleNamespace

import pytest
import torch

from lerobot.datasets.hiva_duration_sidecar import DurationSidecarDataset
from lerobot.processor.converters import batch_to_transition, transition_to_batch
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


class TinyRawDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.meta = SimpleNamespace(camera_keys=[])
        self.num_frames = 2
        self.num_episodes = 1
        self.rows = [
            {"index": 10, "episode_index": 2, "frame_index": 0},
            {"index": 11, "episode_index": 2, "frame_index": 1},
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return {"sample_id": torch.tensor(idx)}

    def get_raw_item(self, idx):
        return self.rows[idx]


def test_duration_sidecar_dataset_loads_frame_labels(tmp_path):
    sidecar_path = tmp_path / "duration_sidecar.jsonl"
    rows = [
        {"dataset_index": 10, "episode_index": 2, "frame_index": 0, "duration_class": 2, "duration_label": 8},
        {"dataset_index": 11, "episode_index": 2, "frame_index": 1, "duration_class": 0, "duration_label": 1},
    ]
    sidecar_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    dataset = DurationSidecarDataset(TinyRawDataset(), sidecar_path)
    sample = dataset[0]

    assert sample["duration_class"].item() == 2
    assert sample["duration_label"].item() == 8

    batched = dataset.__getitems__([0, 1])
    assert [sample["duration_class"].item() for sample in batched] == [2, 0]


def test_duration_class_survives_policy_preprocessor_conversion():
    batch = {
        ACTION: torch.zeros(2, 50, 7),
        "task": ["one", "two"],
        "duration_class": torch.tensor([2, 0]),
        "duration_label": torch.tensor([8, 1]),
    }

    converted = transition_to_batch(batch_to_transition(batch))

    assert torch.equal(converted["duration_class"], batch["duration_class"])
    assert torch.equal(converted["duration_label"], batch["duration_label"])


def test_duration_token_training_forward_uses_duration_class():
    pytest.importorskip("transformers")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    class DummyDurationModel:
        def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
            action_losses = torch.ones_like(actions)
            duration_logits = torch.tensor([[0.0, 0.0, 4.0], [3.0, 0.0, 0.0]], device=actions.device)
            return action_losses, duration_logits

    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    policy.config = SimpleNamespace(
        adapt_to_pi_aloha=False,
        action_feature=SimpleNamespace(shape=(7,)),
        max_action_dim=7,
        use_duration_head=True,
        duration_loss_weight=0.5,
        duration_classes=(1, 3, 8),
    )
    policy.model = DummyDurationModel()
    policy.prepare_images = lambda batch: ([], [])
    policy.prepare_state = lambda batch: batch[OBS_STATE]
    policy.prepare_action = lambda batch: batch[ACTION]

    batch = {
        OBS_STATE: torch.zeros(2, 32),
        ACTION: torch.zeros(2, 50, 7),
        OBS_LANGUAGE_TOKENS: torch.zeros(2, 4, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 4, dtype=torch.bool),
        "duration_class": torch.tensor([2, 0]),
    }

    loss, metrics = SmolVLAPolicy.forward(policy, batch)

    assert torch.isfinite(loss)
    assert metrics["duration_acc"] == 1.0
    assert metrics["duration_target_mean"] == 4.5
    assert metrics["action_loss"] == 1.0


def test_duration_head_execution_horizon_queues_predicted_prefix():
    pytest.importorskip("transformers")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    policy.config = SimpleNamespace(n_action_steps=8)
    policy._queues = {ACTION: deque(maxlen=policy.config.n_action_steps)}

    actions = torch.arange(50 * 7, dtype=torch.float32).reshape(1, 50, 7)
    duration_steps = torch.tensor([3])
    execution_horizon = SmolVLAPolicy._execution_horizon_from_duration(policy, duration_steps, actions)
    policy._queues[ACTION].extend(actions.transpose(0, 1)[:execution_horizon])

    assert execution_horizon == 3
    assert len(policy._queues[ACTION]) == 3


def test_eval_duration_overlay_draws_policy_debug_values():
    from lerobot.scripts.lerobot_eval import _maybe_draw_policy_duration_overlay

    policy = SimpleNamespace(
        config=SimpleNamespace(use_duration_head=True),
        _last_execution_horizon=3,
        _duration_inference_count=7,
    )
    image = torch.zeros(48, 96, 3, dtype=torch.uint8).numpy()

    overlaid = _maybe_draw_policy_duration_overlay(image, policy)

    assert overlaid.shape == image.shape
    assert overlaid[:24].sum() > 0
