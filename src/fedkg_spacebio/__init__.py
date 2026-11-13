"""fedkg-spacebio: a federated knowledge-graph agent for NASA OSDR space biology data.

Subpackages (see CLAUDE.md for rules and README.md for the plan):

- census      : enumerate OSDR studies and build the census table          (phase 1)
- sites       : site registry, manifests, and site definitions             (phase 1)
- harmonize   : ontology, gene-ID and ortholog mapping, drop-rate logging  (phase 2, owned core)
- evaluation  : leave-one-site-out splits, leakage guard, metrics          (phase 3, owned core)
- federation  : FLARE job recipes, aggregation, intermittent-link shim     (phases 3 / 3b)
- policy      : per-site policy-as-code evaluation                         (phase 4, owned core)
- agent       : tools, planner, grounding checker, trace logging           (phase 4)
"""

__version__ = "0.0.1"
