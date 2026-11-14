"""The agent's tool set, assembled in one place (CONTRACTS §7).

Part A: ``sites.list``, ``policy.check``, ``policy.check_plan``, ``results.read``. ``job.submit``,
``job.status`` and ``kg.query`` are added when they are wired in. The names and versions of the
registry used by a run go into its trace's ``run_start.tools``.
"""

from __future__ import annotations

from fedkg_spacebio.agent import tool_policy, tool_results, tool_sites
from fedkg_spacebio.agent.tools import Registry


def default_registry() -> Registry:
    return Registry([tool_sites.TOOL, *tool_policy.TOOLS, tool_results.TOOL])
