# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixtures this subtree shares with the repository-wide conftest.

`tests/conftest.py` defines both of them, but importing it needs the full test
requirements, so a CPU-only host runs this subtree with
`--confcutdir=tests/v1/kv_offload`, which does not load it. Keep these two
definitions in step with the ones there.
"""

import logging

import pytest


@pytest.fixture()
def default_vllm_config():
    """Set a default VllmConfig for tests that reach get_current_vllm_config()."""
    from vllm.config import VllmConfig, set_current_vllm_config

    config = VllmConfig()
    with set_current_vllm_config(config):
        yield config


@pytest.fixture()
def caplog_vllm(monkeypatch, caplog):
    """Capture vLLM's loggers, which do not propagate to the root logger."""
    from vllm.logger import DEFAULT_LOGGING_CONFIG

    monkeypatch.setitem(DEFAULT_LOGGING_CONFIG["loggers"]["vllm"], "propagate", True)
    monkeypatch.setattr(logging.getLogger("vllm"), "propagate", True)
    with caplog.at_level(logging.INFO):
        yield caplog
