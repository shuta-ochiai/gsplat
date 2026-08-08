# SPDX-FileCopyrightText: Copyright 2024 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the functions in the CUDA extension.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

import pytest
import torch
import gsplat

device = torch.device("cuda:0")


def test_mcmc_strategy_positional_constructor():
    from gsplat.strategy import MCMCStrategy

    strategy = MCMCStrategy(123, 4.5, 6, 7, 8, 9, 0.1, True, 0.2, 30.0)

    assert strategy.cap_max == 123
    assert strategy.noise_lr == 4.5
    assert strategy.refine_start_iter == 6
    assert strategy.refine_stop_iter == 7
    assert strategy.noise_injection_stop_iter == 8
    assert strategy.refine_every == 9
    assert strategy.min_opacity == 0.1
    assert strategy.verbose is True
    assert strategy.noise_opacity_t == 0.2
    assert strategy.noise_opacity_k == 30.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
def test_strategy():
    from gsplat.rendering import rasterization
    from gsplat.strategy import DefaultStrategy, MCMCStrategy

    torch.manual_seed(42)

    # Prepare Gaussians
    N = 100
    params = torch.nn.ParameterDict(
        {
            "means": torch.randn(N, 3),
            "scales": torch.rand(N, 3),
            "quats": torch.randn(N, 4),
            "opacities": torch.rand(N),
            "colors": torch.rand(N, 3),
        }
    ).to(device)
    optimizers = {k: torch.optim.Adam([v], lr=1e-3) for k, v in params.items()}

    # A dummy rendering call
    render_colors, render_alphas, info = rasterization(
        means=params["means"],
        quats=params["quats"],  # F.normalize is fused into the kernel
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=params["colors"],
        viewmats=torch.eye(4).unsqueeze(0).to(device),
        Ks=torch.eye(3).unsqueeze(0).to(device),
        width=10,
        height=10,
        packed=False,
    )

    # Test DefaultStrategy
    strategy = DefaultStrategy(verbose=True)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()
    strategy.step_pre_backward(params, optimizers, state, step=600, info=info)
    render_colors.mean().backward(retain_graph=True)
    strategy.step_post_backward(params, optimizers, state, step=600, info=info)

    # Test MCMCStrategy
    strategy = MCMCStrategy(verbose=True)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()
    render_colors.mean().backward(retain_graph=True)
    strategy.step_post_backward(params, optimizers, state, step=600, info=info, lr=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
def test_strategy_requires_grad():
    from gsplat.rendering import rasterization
    from gsplat.strategy import DefaultStrategy, MCMCStrategy

    def assert_consistent_sizes(params):
        sizes = [v.shape[0] for v in params.values()]
        assert all([s == sizes[0] for s in sizes])

    torch.manual_seed(42)

    # Prepare Gaussians
    N = 100
    params = torch.nn.ParameterDict(
        {
            "means": torch.randn(N, 3),
            "scales": torch.rand(N, 3),
            "quats": torch.randn(N, 4),
            "opacities": torch.rand(N),
            "colors": torch.rand(N, 3),
            "non_trainable_features": torch.rand(N, 3),
        }
    ).to(device)
    params["non_trainable_features"].requires_grad = False
    requires_grad_map = {k: v.requires_grad for k, v in params.items()}
    optimizers = {
        k: torch.optim.Adam([v], lr=1e-3) for k, v in params.items() if v.requires_grad
    }

    # A dummy rendering call
    render_colors, render_alphas, info = rasterization(
        means=params["means"],
        quats=params["quats"],  # F.normalize is fused into the kernel
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=params["colors"],
        viewmats=torch.eye(4).unsqueeze(0).to(device),
        Ks=torch.eye(3).unsqueeze(0).to(device),
        width=10,
        height=10,
        packed=False,
    )

    # Test DefaultStrategy
    strategy = DefaultStrategy(verbose=True)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()
    strategy.step_pre_backward(params, optimizers, state, step=600, info=info)
    render_colors.mean().backward(retain_graph=True)
    strategy.step_post_backward(params, optimizers, state, step=600, info=info)
    for k, v in params.items():
        assert v.requires_grad == requires_grad_map[k]
    assert params["non_trainable_features"].grad is None
    assert_consistent_sizes(params)
    # Test MCMCStrategy
    strategy = MCMCStrategy(verbose=True)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()
    render_colors.mean().backward(retain_graph=True)
    strategy.step_post_backward(params, optimizers, state, step=600, info=info, lr=1e-3)
    assert params["non_trainable_features"].grad is None
    for k, v in params.items():
        assert v.requires_grad == requires_grad_map[k]
    assert_consistent_sizes(params)


def test_relocation_target_probs():
    from gsplat.strategy.ops import _relocation_target_probs

    min_opacity = 0.005

    # Weak targets (opacity < 2*min_opacity) get excluded (zero weight);
    # strong ones keep their opacity as weight.
    opacities = torch.tensor([0.001, 0.005, 0.009, 0.011, 0.5, 0.9])
    weak = opacities < 2 * min_opacity
    probs = _relocation_target_probs(opacities, min_opacity)
    assert torch.all(probs[weak] == 0)
    assert torch.equal(probs[~weak], opacities[~weak])

    # All-weak candidate set falls back to plain opacity weighting rather
    # than an all-zero distribution (which torch.multinomial can't sample).
    all_weak = torch.tensor([0.001, 0.002, 0.003, 0.0001])
    fallback_probs = _relocation_target_probs(all_weak, min_opacity)
    assert torch.equal(fallback_probs, all_weak)
    assert torch.any(fallback_probs > 0)

    # [N, 1]-shaped opacities (as stored in params["opacities"]) are flattened.
    opacities_col = opacities.unsqueeze(-1)
    probs_col = _relocation_target_probs(opacities_col, min_opacity)
    assert probs_col.shape == (opacities.shape[0],)
    assert torch.equal(probs_col, probs)

    # multinomial sampling from the excluding distribution never selects a
    # zero-weight (weak) candidate.
    torch.manual_seed(0)
    sampled = torch.multinomial(probs, 2000, replacement=True)
    assert not torch.any(weak[sampled])


if __name__ == "__main__":
    test_strategy()
    test_strategy_requires_grad()
    test_relocation_target_probs()
