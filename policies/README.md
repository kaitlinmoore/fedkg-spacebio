# Testbed policies

**These policies are invented for the testbed.** No mission, agency or data owner declared them.
The sites are simulated over public OSDR data, and the policies exist to test that the policy
check enforces what a site declares and names the rule that decided (CONTRACTS §5).

One folder per policy profile, one YAML file per site, plus the project policy (`project.yaml`).
The schema is `site-policy/1` (CONTRACTS §5.1).

| Profile | What differs |
|---|---|
| `open` | Every site allows per-site summaries (`pol:<site>:1`). |
| `restricted` | SpaceX-16 allows only masked contributions to a secure sum, and adds a differencing rule (`pol:SpaceX-16:2`). Every other file is identical to `open`. |

The worked examples these files are tested against are in CONTRACTS §5.5. The policy test suite
(`tests/test_policy_*.py`) loads these exact files, so changing a rule here changes what the
suite checks.

Site manifests name their site's policy by ID (`policy_id`, CONTRACTS §3). A manifest that
names a policy other than the one loaded for its site is an error, not a silent pick.
