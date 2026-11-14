"""Live smoke tests of both model adapters (``-m live``; skipped by default and without a key).

Each sends one small request with the real Part A tool definitions and checks that the provider
accepts them (including Claude's strict schemas) and that the reply parses. They spend a few
hundred tokens. Keys come from ``ANTHROPIC_API_KEY`` / ``NVIDIA_API_KEY``; the NIM model from
``FEDKG_NIM_MODEL`` (default ``meta/llama-3.3-70b-instruct``).
"""

from __future__ import annotations

import os

import pytest

from fedkg_spacebio.agent import model_claude, model_openai
from fedkg_spacebio.agent.model import UserText, tool_specs
from fedkg_spacebio.agent.toolset import default_registry

pytestmark = pytest.mark.live
PROMPT = "List the ISS clients by calling the sites tool once. Don't answer in text first."


def _check(resp):
    assert resp.stop in ("tool_calls", "end")
    assert resp.served_model
    for call in resp.turn.tool_calls:
        assert call.name in {"sites.list", "policy.check", "policy.check_plan", "results.read"}


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="no ANTHROPIC_API_KEY")
def test_claude_live():
    c = model_claude.ClaudeClient(model_claude.settings(max_tokens=2000, effort="low"))
    _check(c.complete("You are testing a tool interface.", [UserText(PROMPT)],
                      tool_specs(default_registry())))


@pytest.mark.skipif(not os.environ.get("NVIDIA_API_KEY"), reason="no NVIDIA_API_KEY")
def test_nim_live():
    model = os.environ.get("FEDKG_NIM_MODEL", "meta/llama-3.3-70b-instruct")
    c = model_openai.OpenAICompatClient(model_openai.settings(model, max_tokens=500))
    _check(c.complete("You are testing a tool interface.", [UserText(PROMPT)],
                      tool_specs(default_registry())))
