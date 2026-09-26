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
    automotive_populated: bool | None = None,
    finance_populated: bool | None = None,
    finance_first_record_id: str | None = None,
    seed_demo: bool = False,
) -> tuple[str, ...]:
    """Return the startup actions for the databases that are present.

    ``finance_first_record_id`` is accepted and ignored. A value that does
    not start with ``CFPB-`` used to trigger a full fixture reset.

    When ``*_populated`` is supplied, it overrides the ``*_exists`` flag for
    the seeding decision — this lets the caller distinguish "file exists but
    is empty" (created by migrations) from "file has data" (already seeded).
    """
    del finance_first_record_id
    auto_has_data = automotive_populated if automotive_populated is not None else automotive_exists
    fin_has_data = finance_populated if finance_populated is not None else finance_exists
    actions: list[str] = ["init-ops"]
    if seed_demo and not auto_has_data:
        actions.append("seed-automotive")
    else:
        actions.append("init-automotive")
    if seed_demo and not fin_has_data:
        actions.append("seed-finance")
    else:
        actions.append("init-finance")
    return tuple(actions)
