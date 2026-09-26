"""What a container may do on startup.

Missing files get a schema. An explicit demo flag may fill a domain file
that is not there yet. Nothing in this decision resets an existing database,
and a finance record id is not evidence that the warehouse is corrupt.
"""

from __future__ import annotations


def bootstrap_actions(
    *,
    ops_exists: bool,
    automotive_exists: bool,
    finance_exists: bool,
    finance_first_record_id: str | None = None,
    seed_demo: bool = False,
) -> tuple[str, ...]:
    """Return the startup actions for the databases that are present.

    ``finance_first_record_id`` is accepted and ignored. A value that does
    not start with ``CFPB-`` used to trigger a full fixture reset.
    """
    del finance_first_record_id
    actions: list[str] = ["init-ops"]
    if seed_demo and not automotive_exists:
        actions.append("seed-automotive")
    else:
        actions.append("init-automotive")
    if seed_demo and not finance_exists:
        actions.append("seed-finance")
    else:
        actions.append("init-finance")
    return tuple(actions)
