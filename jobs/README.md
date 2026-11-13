# FLARE jobs

Job definitions built with FLARE's Job Recipe API (FLARE 2.9.x). Run them in the FL simulator
from WSL:

```bash
uv run python jobs/<job_name>.py
```

Simulator workspaces go under `$FEDKG_WORKDIR` (set by the `fedkg` shell function; defaults to
`~/fedkg-work` in WSL), not inside the repo. Writes to the Windows drive from WSL can fail.
