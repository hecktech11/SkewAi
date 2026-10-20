# Enhanced vs deterministic intake (synthetic call set)

60 SYNTHETIC calls (`eval/enhanced/calls.jsonl`, built by `build_calls.py`): English, Hindi (Devanagari) and Hinglish, labelled with make / model / year / component and whether the pack kill-switch should fire.

Enhanced mode: **not run (no META_API_KEY configured)**.

| Language | Mode | Make | Model | Year | Component | All 4 slots | Kill-switch recall | False alarms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| en | deterministic | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0 |
| en | enhanced | not run | not run | not run | not run | not run | not run | not run |
| hi | deterministic | 0.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0 |
| hi | enhanced | not run | not run | not run | not run | not run | not run | not run |
| hinglish | deterministic | 1.0 | 1.0 | 1.0 | 0.875 | 0.875 | 0.5 | 0 |
| hinglish | enhanced | not run | not run | not run | not run | not run | not run | not run |
| all | deterministic | 0.667 | 1.0 | 1.0 | 0.625 | 0.625 | 0.5 | 0 |
| all | enhanced | not run | not run | not run | not run | not run | not run | not run |

Slot accuracy is the share of NON-safety calls whose slot exactly matches the label (single customer turn,
no re-asks); on safety calls the existing pipeline escalates first and defers slot capture, so those calls are
scored on kill-switch recall only.
The deterministic path is the unchanged product. Enhanced only fills EMPTY slots with validated values and
can only add a kill-switch (never remove one), so it cannot score lower on a slot the deterministic path got right.

Generated 2026-10-20 21:45 UTC.
