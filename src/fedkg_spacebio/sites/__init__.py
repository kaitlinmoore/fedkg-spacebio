"""Site registry and manifests (phase 1).

Defines the simulated sites (by mission, agency or lab; rationale in SITES.md), loads site
manifests, and exposes each site's data ONLY to that site's own code. The aggregator never
sees raw site data.
"""
