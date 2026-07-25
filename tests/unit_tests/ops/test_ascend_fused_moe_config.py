# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch  # noqa: F401 - load device backends before temporarily mocking torch_npu


def _load_fused_moe_module():
    module_path = (
        Path(__file__).parents[3]
        / "vllm_fl/dispatch/backends/vendor/ascend/impl/fused_moe.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_test_ascend_fused_moe_config", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get("torch_npu")
    sys.modules["torch_npu"] = ModuleType("torch_npu")
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop("torch_npu", None)
        else:
            sys.modules["torch_npu"] = previous
    return module


@pytest.fixture(scope="module")
def fused_moe_module():
    return _load_fused_moe_module()


@pytest.mark.parametrize("value", [None, "", "0", "off", "false", "none"])
def test_parse_moe_gmm_tuning_disabled(fused_moe_module, value):
    assert fused_moe_module._parse_moe_gmm_tuning_mode(value) is None


def test_parse_moe_gmm_tuning_modes(fused_moe_module):
    assert fused_moe_module._parse_moe_gmm_tuning_mode("auto") == "auto"
    assert fused_moe_module._parse_moe_gmm_tuning_mode("2") == 2
    with pytest.raises(ValueError):
        fused_moe_module._parse_moe_gmm_tuning_mode("invalid")
    with pytest.raises(ValueError):
        fused_moe_module._parse_moe_gmm_tuning_mode("-1")


def test_build_moe_gmm_tuning_config(fused_moe_module):
    build = fused_moe_module._build_moe_gmm_tuning_config
    assert build(64, 8, 256, "auto", has_expert_map=False) == [2]
    assert build(256, 8, 256, "auto", has_expert_map=False) == [8]
    assert build(257, 8, 256, "auto", has_expert_map=False) is None
    assert build(1035, 8, 256, "auto", has_expert_map=False) is None
    assert build(64, 8, 256, 4, has_expert_map=False) == [4]
    assert build(4096, 8, 256, 4, has_expert_map=False) is None
    assert build(64, 8, 256, None, has_expert_map=False) is None
    assert build(64, 8, 256, "auto", has_expert_map=True) is None
