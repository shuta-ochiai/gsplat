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

"""Regression coverage for gsplat.strategy.ops.relocate.

relocate() overwrites the *parameter values* of both the donor GSs
(sampled_idxs) and the resurrected/dead GSs (dead_indices) -- but its
optimizer_fn only zeroed Adam's momentum (exp_avg/exp_avg_sq) for
sampled_idxs. A GS flagged dead got there by having its opacity driven down
over many steps, so its momentum was consistently negative; relocate() jumps
the *value* up to a healthy opacity but left that stale negative momentum in
place at dead_indices, so the very next Adam step(s) would drag the freshly
relocated opacity back down -- with no new negative gradient required -- often
re-crossing min_opacity before the next relocate() check 100 steps later.
"""

import math

import pytest
import torch

from gsplat.strategy.ops import relocate

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _binomial_table(n_max: int, device: torch.device) -> torch.Tensor:
    binoms = torch.zeros((n_max, n_max), device=device)
    for n in range(n_max):
        for k in range(n + 1):
            binoms[n, k] = math.comb(n, k)
    return binoms


def _make_params_and_optimizers(opacities: torch.Tensor):
    N = opacities.shape[0]
    torch.manual_seed(0)
    params = {
        "means": torch.nn.Parameter(torch.randn(N, 3, device=device)),
        "scales": torch.nn.Parameter(
            torch.log(torch.rand(N, 3, device=device) * 0.1 + 0.01)
        ),
        "quats": torch.nn.Parameter(torch.randn(N, 4, device=device)),
        "opacities": torch.nn.Parameter(opacities.clone()),
    }
    optimizers = {name: torch.optim.Adam([p], lr=0.05) for name, p in params.items()}
    # Prime each optimizer's state (exp_avg/exp_avg_sq only exist after step()).
    for name, opt in optimizers.items():
        params[name].sum().backward()
        opt.step()
        opt.zero_grad()
    return params, optimizers


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="relocate() depends on the CUDA relocation op"
)
def test_relocate_resets_optimizer_state_for_dead_indices():
    N = 8
    dead_indices = torch.tensor([0, 1], device=device)
    mask = torch.zeros(N, dtype=torch.bool, device=device)
    mask[dead_indices] = True

    opacities = torch.logit(torch.rand(N, device=device) * 0.5 + 0.3)
    params, optimizers = _make_params_and_optimizers(opacities)

    # Hand-plant a large, consistently-negative momentum on the dying GSs,
    # simulating many prior steps of "keep decreasing this opacity" gradient
    # history before they crossed min_opacity and got flagged dead.
    opa_state = optimizers["opacities"].state[params["opacities"]]
    with torch.no_grad():
        opa_state["exp_avg"][dead_indices] = -10.0
        opa_state["exp_avg_sq"][dead_indices] = 10.0
        params["opacities"][dead_indices] = torch.logit(
            torch.tensor(0.001, device=device)
        )

    binoms = _binomial_table(51, device)
    relocate(params, optimizers, state={}, mask=mask, binoms=binoms, min_opacity=0.005)

    new_opacity_param = params["opacities"]
    new_state = optimizers["opacities"].state[new_opacity_param]
    assert torch.all(new_state["exp_avg"][dead_indices] == 0), (
        "relocate() must zero Adam momentum for resurrected GSs (dead_indices), "
        "not just for the donor GSs (sampled_idxs)"
    )
    assert torch.all(new_state["exp_avg_sq"][dead_indices] == 0)
    # Donor-side reset must still hold (this direction already worked pre-fix).
    dead_to_donor = new_opacity_param[dead_indices]
    assert torch.sigmoid(dead_to_donor).min() >= 0.005


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="relocate() depends on the CUDA relocation op"
)
def test_relocate_survives_zero_gradient_steps_after_reset():
    """Behavioral check: with the reset in place, a relocated GS's opacity
    must not collapse back below min_opacity purely from stale Adam momentum
    when no new gradient pushes it down.
    """
    N = 8
    dead_indices = torch.tensor([0, 1], device=device)
    mask = torch.zeros(N, dtype=torch.bool, device=device)
    mask[dead_indices] = True

    opacities = torch.logit(torch.rand(N, device=device) * 0.5 + 0.3)
    params, optimizers = _make_params_and_optimizers(opacities)

    opa_state = optimizers["opacities"].state[params["opacities"]]
    with torch.no_grad():
        opa_state["exp_avg"][dead_indices] = -10.0
        opa_state["exp_avg_sq"][dead_indices] = 10.0
        params["opacities"][dead_indices] = torch.logit(
            torch.tensor(0.001, device=device)
        )

    binoms = _binomial_table(51, device)
    relocate(params, optimizers, state={}, mask=mask, binoms=binoms, min_opacity=0.005)

    opacity_param = params["opacities"]
    opt = optimizers["opacities"]
    for _ in range(20):
        (opacity_param.sum() * 0.0).backward()  # zero gradient, momentum-only dynamics
        opt.step()
        opt.zero_grad()

    final_opacity = torch.sigmoid(opacity_param[dead_indices])
    assert torch.all(final_opacity >= 0.005), (
        f"resurrected GS opacity collapsed back below min_opacity "
        f"({final_opacity.tolist()}) from stale momentum alone, with zero new "
        f"gradient signal"
    )
