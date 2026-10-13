"""Post-close enrichment backfill (audit 4.1).

When an enrichment agent fails mid-contact, the case closes with partial
evidence. This job re-runs the investigator against the persisted slot
frame after close, patches the case row (cluster, similar count,
investigation link), and ledgers the late brief with
``provenance=backfilled`` so Qubot can tell it apart from contact-time
evidence. Leased + retried with backoff like every other job type.
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.data.warehouse import ops_con


def backfill_brief(interaction_id: str, *, case_id: str | None = None) -> dict[str, Any]:
    """Recompute the investigation brief for a closed contact. Returns a report."""
    iid = (interaction_id or "").strip()
    if not iid:
        return {"ok": False, "error": "interaction_id required"}
    with ops_con(read_only=True) as con:
        try:
            cur = con.execute(
                """
                SELECT pack_id, entity_1, entity_2, entity_3, category,
                       description
                FROM interactions WHERE interaction_id = ?
                """,
                [iid],
            )
            row = cur.fetchone()
        except Exception as e:
            return {"ok": False, "error": f"interaction read failed: {e}"}
    if not row:
        return {"ok": False, "error": "interaction not found"}
    pack_id = row[0]
    slots = {
        "entity_1": row[1],
        "entity_2": row[2],
        "entity_3": row[3],
        "category": row[4],
        "description": row[5],
    }
    from src.agents.base import InteractionContext
    from src.agents.investigator import InvestigatorAgent
    from src.domains.loader import load_pack

    try:
        pack = load_pack(pack_id)
    except Exception as e:
        return {"ok": False, "error": f"pack load failed: {e}"}
    ctx = InteractionContext(interaction_id=iid, pack=pack, channel="backfill")
    ctx.slots.update({k: v for k, v in slots.items() if v})

    def _run_in_new_loop() -> Any:
        # Coroutine is created inside, so a running-loop RuntimeError can
        # never strand a never-awaited coroutine (no warnings, no leaks).

        async def _inner() -> Any:
            return await InvestigatorAgent(ctx).run()

        return asyncio.run(_inner())

    try:
        asyncio.get_running_loop()
        in_loop = True
    except RuntimeError:
        in_loop = False
    try:
        if in_loop:
            import concurrent.futures as _fut

            with _fut.ThreadPoolExecutor(max_workers=1) as ex:
                result = ex.submit(_run_in_new_loop).result()
        else:
            result = _run_in_new_loop()
    except Exception as e:
        return {"ok": False, "error": f"investigator failed: {e}", "preserved": True}
    if not isinstance(result, dict):
        return {"ok": False, "outcome": "failed", "preserved": True, "interaction_id": iid}
    outcome = str(result.get("outcome") or "")
    if result.get("skipped") or (outcome and outcome != "completed"):
        return {
            "ok": False,
            "outcome": outcome or "failed",
            "preserved": True,
            "interaction_id": iid,
        }
    brief = ctx.investigation_brief or {}
    cluster_id = brief.get("cluster_id")
    similar_n = int(brief.get("similar_record_count") or 0)
    target_case = case_id
    if not target_case:
        with ops_con(read_only=True) as con:
            try:
                found = con.execute(
                    "SELECT case_id FROM cases WHERE interaction_id = ? LIMIT 1",
                    [iid],
                ).fetchone()
                target_case = str(found[0]) if found else None
            except Exception:
                target_case = None
    patched = False
    investigation_id: str | None = None
    if target_case:
        try:
            from src.frontline.live_intercept import open_or_link_investigation

            if cluster_id is not None:
                try:
                    investigation_id = open_or_link_investigation(
                        pack_id=pack_id,
                        cluster_id=int(cluster_id),
                        title=f"Backfilled cluster {cluster_id}",
                    )
                except Exception:
                    investigation_id = None
            with ops_con() as con:
                con.execute(
                    """
                    UPDATE cases
                    SET cluster_match_id = COALESCE(?, cluster_match_id),
                        similar_record_count = CASE
                            WHEN ? > 0 THEN ?
                            ELSE similar_record_count
                        END,
                        investigation_id = COALESCE(?, investigation_id)
                    WHERE case_id = ?
                    """,
                    [cluster_id, similar_n, similar_n, investigation_id, target_case],
                )
            patched = True
        except Exception as e:
            return {"ok": False, "error": f"case patch failed: {e}"}
    try:
        from src.ledger import AgentAction, record_action

        record_action(AgentAction(
            interaction_id=iid,
            agent="investigator",
            action_type="brief_written",
            input_summary="reenrich backfill after enrichment failure",
            output_summary=(
                f"provenance=backfilled cluster_id={cluster_id} "
                f"similar={similar_n} case={target_case}"
            ),
            evidence_ids=[str(cluster_id)] if cluster_id is not None else [],
            case_id=target_case,
        ))
    except Exception:
        pass
    return {
        "ok": True,
        "interaction_id": iid,
        "case_id": target_case,
        "cluster_id": cluster_id,
        "similar_record_count": similar_n,
        "investigation_id": investigation_id,
        "case_patched": patched,
        "provenance": "backfilled",
    }


__all__ = ["backfill_brief"]
