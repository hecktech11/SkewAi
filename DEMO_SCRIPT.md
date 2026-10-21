# Skew AI + Meta AI layer: 2–3 minute demo script

> **Mission: help vehicle manufacturers reduce road accidents by recording
> vehicle faults from real customer contacts and predicting them early,
> before they become crashes, fires or injuries.**

Setup (one command, local, nothing deployed):

```bash
bash scripts/judge_setup.sh          # venv, deps, dashboard build, demo DBs, India scenario, serves :8000
# then open http://127.0.0.1:8000/ui/#demo, Sign in → "Have an operator API key?" → paste the key the script printed
```

Without a Meta key everything below runs on the deterministic path, and the badge
reads **Deterministic**. With `META_API_KEY` (or `MODEL_API_KEY`) set the
badge reads **Meta AI enhanced** and the Meta model tags light up.
No numbers in this script are made up. Backtest figures come from
`reports/nhtsa/backtest_safety_2015_2024.md`, and eval figures come from
`eval/enhanced/REPORT.md`.

---

## 0:00–0:20 The problem (title card, then the `#demo` page header)

Voice-over:
"Every recall starts as a few customer calls that nobody connects. In India
most of those calls are in Hindi or Hinglish. Skew AI records each call as a
structured fault, clusters calls across a fleet, and warns the manufacturer
early. Today we'll add Meta's models on top of that without letting them
override any safety rule."

On screen: `#demo` page, mission banner, the **AI mode badge** in the top bar.

## 0:20–0:55 Screen 1: a Hindi/Hinglish call plus a photo

- Click **Run seeded India scenario** (or show the call widget, with the speech
  language set to **हिन्दी / Hinglish (hi-IN)**).
- Transcript: *"Namaste. Meri Honda CR-V 2019 ke brakes fail ho gaye, pedal
  neeche tak chala jata hai aur grinding ki awaaz aati hai."*
- **The key moment, shown side by side:**
  - *Deterministic path* (rules only): the call completes and opens a case at
    **Medium**. The English-only kill-switch lexicon does not recognise
    Hinglish "brakes fail ho gaye".
  - *Meta AI enhanced* (Muse Spark normalises Hinglish to English, then the
    **existing** pack kill-switch runs on the translation): the case goes to
    **Critical** with **"do not drive"** guidance. The AI did not decide that
    the call was safety-critical. The existing rule did, once it could read the
    words. AI can only raise severity, never lower it.
- A photo of the brake rotor is attached to the case. In the demo it is a
  labelled **SYNTHETIC** image, never a real customer photo.

Tags on screen: `Muse Spark 1.3` (language normalisation), `deterministic rules`.

## 0:55–1:25 Screen 2: the case with photo evidence

- Open the case from the demo link (Case queue → **Photo evidence** panel).
- Click **Analyze**: **SAM 3.1** returns region boxes that are drawn as an
  overlay on the photo. **Muse Spark** (vision) adds a short description and
  a component guess.
- Call out the safety rules: a vision finding can only *add* a flag; the photo
  is validated (type, size, magic bytes), stored on the box and linked to the
  case and its cluster/investigation; it appears in the DSR erase.

Tags: `SAM 3.1`, `Muse Spark 1.3 (vision)`.

## 1:25–1:55 Screen 3: the grounded brief

- Issue desk → the investigation the call joined (HONDA · SERVICE BRAKES).
- **Grounded brief** panel: every claim cites real record IDs or recall
  advisory IDs. Each claim passes the existing **Qubot auditor**; claims that
  fail are dropped and counted, not shown.
- **Safety risk chip** (High / Elevated / Normal): computed from the
  safety-critical component tier and the crash/fire/injury mentions in the real
  records. It is deterministic, not AI.
- **Recommended action**: the matched recall remedy comes first. Then come the
  deterministic inspection playbook steps for the component. AI-suggested
  steps are labelled "requires engineering review".

Tags: `Muse Spark 1.3` (brief wording), `Qubot audited`.

## 1:55–2:35 Screen 4: the early-warning board

- Early-warning board → **Safety-risk board**: open investigations ranked by
  safety risk first, then by case count.
- **Backtest headline** (real NHTSA data, Honda/Toyota/Ford/Hyundai/Kia, model
  years 2015–2024; read live from `reports/nhtsa/backtest_safety.json`):
  **535 of 1,172 recall campaigns (45.6%)** had a flag episode on the same
  make + component within the 52 weeks before NHTSA received the recall report.
  The median lead was **11 weeks** (mean 15.9), scored only on complaints that
  had arrived by that date. Episode precision is 0.64 (36% of flag episodes
  had no later recall). In those windows NHTSA received **1,379 complaints
  marked CRASH, 712 marked FIRE and 882 with injuries** on the flagged slices.
- Say the caveat out loud: "These are complaints filed during the window
  between our flag and the recall. They show correlation and potential
  exposure. We do not claim lives saved."

## 2:35–2:50 Close

"Meta's models make the system hear Hindi, see the part, and write the brief.
The deterministic safety rules still have the final say. Without a key it
behaves exactly as before."

Show: `eval/enhanced/REPORT.md`. On 60 labelled synthetic calls (en/hi/
hinglish), the deterministic path recalls the kill-switch on 100% of the
English calls but on 0% of the Hindi calls; the enhanced column reads
"not run" until a key is set.
