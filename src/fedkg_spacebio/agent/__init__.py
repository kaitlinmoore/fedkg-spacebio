"""The agent (phase 4).

Tools (KG query via mcp-genelab, site registry, policy check, FLARE job submit/monitor, results
reader), the planner, trace logging, and the grounding checker (owned core): a claim is supported
only if the ID it cites appeared in this trace's own tool results. Never fabricate IDs.
"""
