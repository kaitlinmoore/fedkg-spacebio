"""Inspect, verify or reset a profile's release log, from the command line (CONTRACTS §5.3).

    uv run python -m fedkg_spacebio.policy.log show   --profile restricted
    uv run python -m fedkg_spacebio.policy.log verify --profile restricted
    uv run python -m fedkg_spacebio.policy.log reset  --profile restricted --reason "rehearsal 2"

- ``show`` lists the profile's log history (start and switch events) and the active log's
  releases: what was released, to which clients, over how many genes. No output values: the log
  has none.
- ``verify`` checks the hash chain, every contributor file, and that every decision record made
  against the log names a head that is still in it (catches entries removed from the end).
  Exit 1 if anything doesn't check out.
- ``reset`` starts ``<profile>-<n+1>``. It needs ``--reason``, which is recorded with the old
  log's head in ``logs/index.jsonl``. Nothing is deleted.

Exit status: 0 ok, 1 integrity failure, 2 usage error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fedkg_spacebio.policy.release_log import (
    LogIntegrityError,
    active_log,
    cross_check_decisions,
    default_root,
    read_index,
    reset_log,
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m fedkg_spacebio.policy.log",
                                description="Show, verify or reset a release log.")
    p.add_argument("command", choices=["show", "verify", "reset"])
    p.add_argument("--profile", required=True)
    p.add_argument("--root", type=Path, help="policy root (default: $FEDKG_WORKDIR/policy)")
    p.add_argument("--reason", help="why the log is reset (required for reset)")
    return p


def _show(profile: str, root: Path) -> int:
    for e in read_index(root):
        if e.get("profile") != profile:
            continue
        if e["event"] == "start":
            print(f"{e['ts']}  start   {e['log_id']}")
        else:
            print(f"{e['ts']}  switch  {e['previous_log_id']} -> {e['log_id']}  "
                  f"(head {e['previous_head']}; {e['reason']})")
    log = active_log(profile, root=root, create=False)
    entries = log.entries()
    print(f"active log: {log.log_id}, {len(entries)} release{'s' if len(entries) != 1 else ''}")
    for e in entries:
        n_genes = len(log.contributors(e))
        kinds = sorted(set(e["outputs"].values()))
        print(f"  {e['seq']:>3}  {e['release_id']}  {e['data_key']['task']}/"
              f"{e['data_key']['stratum']}  {e['counts_variant']}  {n_genes} genes  "
              f"{', '.join(e['clients'])}  [{', '.join(kinds)}]")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    root = args.root or default_root()
    try:
        if args.command == "show":
            return _show(args.profile, root)
        if args.command == "verify":
            log = active_log(args.profile, root=root, create=False)
            n = log.verify()
            k = cross_check_decisions(log)
            print(f"{log.log_id}: {n} entr{'y' if n == 1 else 'ies'} and their contributor "
                  f"files check out; {k} decision record{'s' if k != 1 else ''} consistent")
            return 0
        if not (args.reason or "").strip():
            print("reset needs --reason (it is recorded with the old log's head)",
                  file=sys.stderr)
            return 2
        new = reset_log(args.profile, args.reason, root=root)
        print(f"started {new.log_id}; the previous log is kept")
        return 0
    except LogIntegrityError as exc:
        print(f"integrity check failed: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
