# Recall lead time

Complaints are counted by `received_at` only, as of each recall's report-received date.
A hit needs a flagged week at least one week before that date.
Misses and flags with no later recall stay in this report.

Pack `recall_lead_sample`. Hits 2. Misses 1. Unconfirmed flags 3.

## Recalls

| Advisory | Issued | Slice | First flagged week | Lead weeks | Result |
| --- | --- | --- | --- | --- | --- |
| 15V710000|2016|FORD|E-450|ELECTRICAL SYSTEM | 2015-10-27 | FORD E-450 ELECTRICAL SYSTEM | 2015-W32 | 12 | hit |
| 17V168000|2017|FORD|MUSTANG|LATCHES/LOCKS/LINKAGES | 2017-03-13 | FORD MUSTANG LATCHES/LOCKS/LINKAGES |  | 0 | miss |
| 19V060000|2015|HONDA|ACCORD|FUEL SYSTEM | 2019-01-29 | HONDA ACCORD FUEL SYSTEM | 2019-W01 | 4 | hit |

## Misses

- 17V168000|2017|FORD|MUSTANG|LATCHES/LOCKS/LINKAGES issued 2017-03-13 (FORD LATCHES/LOCKS/LINKAGES): miss

## Unconfirmed flags

| Week | Slice | Count | Baseline | z | Status |
| --- | --- | --- | --- | --- | --- |
| 2022-W42 | FORD LANE DEPARTURE | 6 | 0.54 | 5.70 | unconfirmed |
| 2023-W21 | FORD LANE DEPARTURE | 6 | 0.54 | 5.70 | unconfirmed |
| 2023-W52 | FORD LANE DEPARTURE | 6 | 0.54 | 5.70 | unconfirmed |

## How to read this file

Complaint rows are counted on `received_at` (NHTSA DATEA).
Recall `issued_at` is RCDATE, the Part 573 report-received date.
A hit is the first flagged week of the same category and make, at least one week before `issued_at`.
The Honda and Ford sample in `eval/frontline/fixtures/recall_lead/` uses real campaign numbers. It is not a five-make download.
Regenerate that sample without touching `data/domains`:

```bash
DOMAIN_DB_PATH=/tmp/skew-recall-lead python -m scripts.recall_lead_sample
```

A larger public window is manual. Point `DOMAIN_DB_PATH` at a side
directory so the pilot warehouse is not replaced:

```bash
DOMAIN_DB_PATH=/tmp/skew-nhtsa make ingest-nhtsa N=200000 MAKES=HONDA,TOYOTA,FORD,HYUNDAI,KIA YEAR_MIN=2015 YEAR_MAX=2024
DOMAIN_DB_PATH=/tmp/skew-nhtsa make ingest-recalls MAKES=HONDA,TOYOTA,FORD,HYUNDAI,KIA YEAR_MIN=2015 YEAR_MAX=2024 RCL_TYPE=V
DOMAIN_DB_PATH=/tmp/skew-nhtsa python -m scripts.recall_lead_report --pack automotive_nhtsa --out reports/nhtsa/recall_lead_time_2015_2024.md
```
