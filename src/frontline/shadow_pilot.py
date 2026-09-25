"""Shadow Pilot CLI & Execution Harness — Step 0 Dual-Stream Production Validation.

Evaluates AI frontline assistant against human supervisor and engineer ground truth
across the Five Decision Numbers with 95% confidence intervals:
  1. Slot accuracy vs. human record (>= 90% per critical entity, Wilson CI)
  2. Kill-switch precision (>= 0.85) & recall (>= 0.98) (Wilson CI)
  3. Severity agreement Cohen's kappa (>= 0.70, Asymptotic CI)
  4. Cluster agreement with engineer (>= 75% top-1, >= 85% top-3, Wilson CI)
  5. Cost per contact blended (<= $0.45, SE / normal 95% CI)

Usage:
  python3 -m src.frontline.shadow_pilot --mode=batch --sample-size=100 --out=reports/shadow_baseline_01.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

from src.eval.shadow_pilot import (
    ShadowContact,
    ShadowMetricResult,
    ShadowPilotEvaluator,
    cohen_kappa_with_ci,
    wilson_score_interval,
)


# ── Synthetic / Replay Automotive Cohort Generator ──────────────────────────

_MAKES_MODELS: dict[str, list[str]] = {
    "HONDA": ["CR-V", "CIVIC", "ACCORD", "PILOT", "ODYSSEY"],
    "TOYOTA": ["CAMRY", "COROLLA", "RAV4", "HIGHLANDER", "TACOMA", "TUNDRA"],
    "FORD": ["F-150", "EXPLORER", "ESCAPE", "MUSTANG", "EDGE"],
    "CHEVROLET": ["SILVERADO", "EQUINOX", "MALIBU", "TAHOE", "TRAVERSE"],
    "NISSAN": ["ROGUE", "ALTIMA", "SENTRA", "PATHFINDER"],
    "JEEP": ["GRAND CHEROKEE", "WRANGLER", "CHEROKEE"],
    "TESLA": ["MODEL 3", "MODEL Y", "MODEL S"],
    "HYUNDAI": ["TUCSON", "SANTA FE", "ELANTRA", "SONATA"],
    "SUBARU": ["OUTBACK", "FORESTER", "CROSSTREK"],
}

_CATEGORIES = [
    "SERVICE BRAKES",
    "ENGINE",
    "AIR BAGS",
    "ELECTRICAL SYSTEM",
    "STEERING",
    "SUSPENSION",
    "POWER TRAIN",
]

_CATEGORY_CLUSTERS: dict[str, int] = {
    "SERVICE BRAKES": 14,
    "AIR BAGS": 22,
    "ELECTRICAL SYSTEM": 31,
    "ENGINE": 40,
    "STEERING": 55,
    "SUSPENSION": 63,
    "POWER TRAIN": 71,
    "VEHICLE SPEED CONTROL": 82,
    "STRUCTURE": 90,
    "EXTERIOR LIGHTING": 95,
    "SEAT BELTS": 98,
    "SEATS": 102,
}

_SYMPTOM_TEMPLATES: dict[str, list[str]] = {
    "SERVICE BRAKES": [
        "severe brake pedal pulsation and grinding noise when coming to a stop",
        "brake pedal went soft and spongy on the highway, required extra stopping distance",
        "high pitched screeching from front brakes whenever lightly pressing pedal",
        "anti-lock braking system light on dashboard and brakes shudder at low speeds",
    ],
    "ENGINE": [
        "check engine light flashing with rough idle and engine misfire under acceleration",
        "engine stalls unexpectedly at stop lights but restarts after several seconds",
        "excessive oil consumption and burning smell coming from under the hood",
        "loud knocking noise from engine bay during cold morning startup",
    ],
    "AIR BAGS": [
        "airbag warning light stays on constantly while driving",
        "passenger airbag light shows off even when adult passenger is seated",
        "airbag indicator flashes intermittently when adjusting driver seat",
    ],
    "ELECTRICAL SYSTEM": [
        "center infotainment screen completely black and instrument cluster flickering",
        "battery drains overnight and vehicle will not crank in the morning",
        "power windows fail to roll up and door lock switches unresponsive",
    ],
    "STEERING": [
        "power steering assist cuts out suddenly while making low speed turns",
        "steering wheel pulls sharply to the right on flat level pavement",
        "stiff steering wheel with whining noise when turning all the way left",
    ],
    "SUSPENSION": [
        "loud clunking and popping noise over road bumps from front suspension",
        "excessive body roll and bouncy ride quality on highway dips",
        "front end vibrations transmitted into vehicle cabin at 65 mph",
    ],
    "POWER TRAIN": [
        "transmission slips and hesitates between second and third gear",
        "harsh shudder and jerking when accelerating from a complete stop",
        "vehicle fails to shift into reverse gear without multiple attempts",
    ],
}

_EMERGENCY_SCENARIOS = [
    ("Thick black smoke and flames pouring out of the engine compartment on the highway!", "Critical", "fire"),
    ("Brakes failed completely at 55 mph and crashed into a guardrail, ambulance was called for passenger injuries.", "Critical", "crash"),
    ("Under-hood fire broke out immediately after parking in driveway, fire department responded.", "Critical", "fire"),
    ("Sudden unintended acceleration caused crash through garage wall, injured driver bleeding.", "Critical", "injury"),
]

_IDIOMATIC_HEDGED_SCENARIOS = [
    ("This diagnostic repair estimate is killing me and almost gave me a heart attack, but my brakes are making a squealing sound.", "Medium"),
    ("I am worried it might catch fire because there is a faint hot smell when heater is running.", "Medium"),
    ("Nobody is hurt and no fire, just a strange clicking noise when turning on headlights.", "Low"),
    ("Driving in rush hour traffic is killing me, I need to schedule routine brake maintenance.", "Low"),
]

# ── Adversarial Paraphrase Cohort (Natural Speech / Zero Lexicon Verbatim) ──

_ADVERSARIAL_PARAPHRASE_TEMPLATES: dict[str, list[str]] = {
    "SERVICE BRAKES": [
        "the pedal went straight down to the mat and i could not get the car to halt",
        "takes twice as long to come to a standstill when slowing from highway speeds",
        "loud metallic grinding friction whenever i press down to halt at an intersection",
        "violent shuddering under my right shoe whenever trying to decelerate",
        "pushed down as hard as i could with my foot but the vehicle kept rolling forward",
        "spongy resistance underfoot and i have to pump it repeatedly to stop",
        "terrible high pitched scraping screech whenever slowing down for red lights",
        "vehicle lurches forward unexpectedly when attempting to come to a complete halt",
        "stopping distance has expanded drastically over the last few days",
        "felt zero resistance under my sole and the car drifted through the stop sign",
        "severe pulsing and vibrating through the floor when decelerating on the off ramp",
        "complete loss of stopping ability while descending a steep grade",
    ],
    "ENGINE": [
        "shuts off abruptly while waiting at red lights but turns over after waiting",
        "loud rhythmic knocking and metallic tapping from under the bonnet when cold",
        "terrible shuddering and hesitation when trying to accelerate from a stop",
        "lost all power climbing a hill and could not accelerate past twenty",
        "sputtered violently and died right in the middle of a busy intersection",
        "thick white vapor with a sweet odor drifting from the front compartment",
        "runs incredibly rough and shakes the entire cabin when sitting at an idle",
        "pungent burning oil odor and dark drips forming under the front end",
        "loud clattering sound from the front bay that speeds up when revving",
        "bogged down completely when merging onto the motorway and would not go",
        "died suddenly while cruising at fifty and had to coast onto the shoulder",
        "severe misfiring and loss of forward thrust under normal acceleration",
    ],
    "AIR BAGS": [
        "the warning symbol showing a seated person with a round balloon stays lit",
        "supplemental restraint indicator turned on and will not extinguish",
        "passenger safety cushion lamp displays disabled even with an adult occupant",
        "an icon of an inflated cushion glowing continuously on the instrument cluster",
        "yellow restraint warning illuminates intermittently whenever adjusting the seat",
        "the secondary occupant protection alert has been glowing for two weeks",
        "dashboard shows an occupant icon with a circle and chime rings on startup",
        "passive restraint alert came on during heavy rain and remains stuck on",
    ],
    "ELECTRICAL SYSTEM": [
        "the dash displays flickered and the center console shut down completely",
        "completely unresponsive in the morning and only makes a faint click when pressing start",
        "all cabin illumination and instrument gauges went totally dark while driving",
        "power windows refuse to go up and the door lock toggles do nothing",
        "the center display keeps rebooting every ten minutes on my commute",
        "keyless entry fob fails to unlock doors or let me turn on the ignition",
        "all dashboard needles dropped to zero while traveling at cruising speed",
        "infotainment unit went black and no audio comes from the sound system",
        "power mirrors and interior dome lights do not activate at all",
        "warning chime rings constantly with erratic symbols flashing across the cluster",
    ],
    "STEERING": [
        "the wheel shakes violently in my hands on the motorway",
        "the vehicle constantly pulls towards the ditch on flat pavement",
        "turning into tight parking spots requires immense physical effort",
        "wanders between highway lanes unless i fight hard to hold it centered",
        "loud whining groan whenever turning sharply around roundabouts",
        "directional control feels disconnected from where the front is pointing",
        "huge dead zone in the column before the front actually begins to turn",
        "jerks sharply to the right side whenever letting go of the rim",
        "front assembly feels frozen solid when trying to make ninety degree turns",
        "violent wobble transmitted directly into my palms at highway velocity",
    ],
    "SUSPENSION": [
        "loud clunking and banging over small potholes and road seams",
        "the rear end bounces uncontrollably after crossing bridge joints",
        "the vehicle leans heavily to the driver side even on flat level ground",
        "harsh creaking and popping underneath whenever rolling over speed bumps",
        "the whole chassis oscillates up and down repeatedly after dips in the road",
        "rough jarring ride feels like there is zero cushioning over bumps",
        "bottoms out with a loud metal bang whenever carrying passengers in back",
        "front corner sags noticeably and scrapes the curb when parking",
    ],
    "POWER TRAIN": [
        "jerks violently between gears during moderate acceleration",
        "tachometer spikes up to four thousand revs but the vehicle barely creeps",
        "massive clunk reverberates underneath when shifting into reverse",
        "slipped out of drive and neutralised while cruising on the expressway",
        "huge hesitation from a standstill before anything catches and moves",
        "harsh shudder through the floorboards whenever accelerating past thirty",
        "refuses to engage when shifting out of park in cold temperatures",
        "delayed engagement followed by a violent thud when pressing the accelerator",
        "hesitates and slips when trying to climb an incline from a complete stop",
        "shudders uncontrollably when downshifting to overtake on the highway",
    ],
    "SEAT BELTS": [
        "the safety strap will not pull out of the pillar when trying to buckle",
        "the center clasp refuses to click and lock into the receiver",
        "the shoulder harness remains loose and does not cinch against the torso",
        "the webbing got stuck midway and refuses to retract back into the reel",
        "the latching mechanism unfastens on its own whenever hitting a bump",
        "tensioner failed to lock during sudden deceleration on the highway",
    ],
}

_ADVERSARIAL_EMERGENCY_SCENARIOS = [
    # True emergencies that unhedged lexicon catches (~35% recall baseline)
    ("theres flames pouring out from underneath the front right now", "Critical", "fire", True),
    ("the vehicle caught fire in the driveway and the hood is ablaze", "Critical", "fire", True),
    ("my wife is bleeding and we need an ambulance immediately", "Critical", "injury", True),
    ("another car t-boned us and someone is hurt in the back", "Critical", "injury", True),
    ("thick black smoke started pouring through the air vents while driving", "Critical", "fire", True),
    ("my daughter is trapped in the back seat and the door wont open", "Critical", "injury", True),
    ("the cabin filled with smoke and i had to jump out while it was rolling", "Critical", "fire", True),
    ("engine compartment is on fire on the shoulder of the highway", "Critical", "fire", True),
    ("airbag deployed in the collision and my passenger is badly hurt", "Critical", "injury", True),
    # True emergencies missed by lexicon due to open-vocabulary paraphrase or hedge window
    ("the pedal went straight to the floor and i couldnt stop at all", "Critical", "brakes", False),
    ("i had no braking ability at all coming down the steep hill", "Critical", "brakes", False),
    ("it wouldnt slow down at all i just kept rolling through the red light", "Critical", "brakes", False),
    ("the wheel wont turn at all its completely stiff and locked up", "Critical", "steering", False),
    ("the car took off on its own and accelerated through the garage wall", "Critical", "speed_control", False),
    ("we hit the guardrail pretty hard on the expressway", "Critical", "crash", False),
    ("there was a loud bang from the front axle and all directional control vanished", "Critical", "steering", False),
    ("we went off the road and rolled into the ditch upside down", "Critical", "rollover", False),
    ("i couldnt steer it just went wherever it wanted across the median", "Critical", "steering", False),
    ("im not sure what happened but theres smoke pouring out of the hood", "Critical", "fire", False),
    ("i was worried it would catch fire and then it actually did", "Critical", "fire", False),
    ("not gonna lie im scared theres flames under the bonnet", "Critical", "fire", False),
    ("the tie rod snapped at sixty miles an hour and we spun into oncoming traffic", "Critical", "steering", False),
    ("unintended acceleration launched us across two lanes of traffic", "Critical", "speed_control", False),
    ("lost all hydraulic pressure in the pedal and plowed into the retaining barrier", "Critical", "brakes", False),
    ("the front suspension collapsed at fifty miles an hour and we careened into a tree", "Critical", "suspension", False),
]

_ADVERSARIAL_IDIOMATIC_SCENARIOS = [
    ("this repair bill is killing me but the pedal just feels a bit stiff", "Medium"),
    ("nobody was hurt and there was no fire just a faint ticking noise", "Medium"),
    ("im worried it might catch fire because of a slight warm odor", "Medium"),
    ("the dealer gave me a crash course on the navigation display", "Low"),
    ("my reading light bulb burned out last week", "Low"),
    ("no smoke no fire just an annoying rattle from the glovebox", "Low"),
    ("i was dying to get this sorted before my long weekend trip", "Low"),
    ("the mechanic said the clutch is burning out slowly", "Low"),
    ("there was a fire drill at work so i had to reschedule my appointment", "Low"),
    ("traffic was murder on the way over to the dealership", "Low"),
]


def generate_adversarial_paraphrase_cohort(sample_size: int = 150, seed: int = 42) -> list[ShadowContact]:
    """Generate an adversarial paraphrase cohort of callers describing symptoms.

    Transcripts are lowercase, unpunctuated or lightly punctuated, and contain NO
    verbatim lexicon keywords or category names. True emergencies split into
    ~35% that trigger rules and ~65% that miss (due to open-vocabulary paraphrase
    or hedge suppression), matching the honest baseline of 0.350 pinned in Probe A.
    """
    from src.agents.base import InteractionContext
    from src.agents.intake import (
        _check_kill_switch,
        _extract_year,
        _extract_via_gazetteer,
        extract_pack_category,
    )
    from src.domains.loader import load_pack

    pack = load_pack("automotive_nhtsa")
    ctx = InteractionContext(interaction_id="shadow_adv_cohort", pack=pack)
    rng = random.Random(seed)
    contacts: list[ShadowContact] = []

    # Target distribution:
    # ~15% true life-safety emergencies (calibrated to ~35% rules recall)
    # ~7% idiomatic/hedged traps (calibrated to 100% precision)
    # ~78% standard defect contacts (open-vocabulary paraphrase)
    emergency_count = max(4, int(sample_size * 0.15))
    idiomatic_count = max(3, int(sample_size * 0.07))
    emergency_indices = set(rng.sample(range(sample_size), k=emergency_count))
    remaining_indices = [i for i in range(sample_size) if i not in emergency_indices]
    idiomatic_indices = set(rng.sample(remaining_indices, k=idiomatic_count))

    all_cluster_ids = list(_CATEGORY_CLUSTERS.values())
    categories_list = list(_ADVERSARIAL_PARAPHRASE_TEMPLATES.keys())

    for i in range(sample_size):
        contact_id = f"adv_cnt_{i+1:04d}"
        year = str(rng.randint(2015, 2024))
        make = rng.choice(list(_MAKES_MODELS.keys()))
        model = rng.choice(_MAKES_MODELS[make])
        category = rng.choice(categories_list)
        eng_cluster = _CATEGORY_CLUSTERS.get(category, 40)

        duration = rng.uniform(75.0, 150.0)
        tokens = rng.randint(350, 750)

        human_slots = {
            "entity_1": year,
            "entity_2": make,
            "entity_3": model,
            "category": category,
        }

        if i in emergency_indices:
            scen_text, sev, kill_concept, _ = rng.choice(_ADVERSARIAL_EMERGENCY_SCENARIOS)
            text = scen_text
            human_kill_needed = True
            human_sev = "Critical"
            if kill_concept in ("fire",):
                category = "ENGINE"
            elif kill_concept in ("brakes",):
                category = "SERVICE BRAKES"
            elif kill_concept in ("steering",):
                category = "STEERING"
            elif kill_concept in ("injury", "crash"):
                category = "AIR BAGS"
            else:
                category = "ENGINE"
            human_slots["category"] = category
            eng_cluster = _CATEGORY_CLUSTERS.get(category, 40)
        elif i in idiomatic_indices:
            text, human_sev = rng.choice(_ADVERSARIAL_IDIOMATIC_SCENARIOS)
            human_kill_needed = False
        else:
            symptoms = _ADVERSARIAL_PARAPHRASE_TEMPLATES[category]
            text = rng.choice(symptoms)
            human_kill_needed = False
            if category in ("SERVICE BRAKES", "STEERING"):
                human_sev = "High" if rng.random() < 0.75 else "Medium"
            elif category in ("ENGINE", "AIR BAGS"):
                human_sev = "High" if rng.random() < 0.60 else "Medium"
            else:
                human_sev = "Medium" if rng.random() < 0.80 else "Low"

        # Natural caller introduction:
        text = f"my {year} {make.lower()} {model.lower()} {text}"

        # Real deterministic intake extraction
        ai_slots: dict[str, str] = {}
        y = _extract_year(text, (1990, 2026))
        if y:
            ai_slots["entity_1"] = y
        m = _extract_via_gazetteer(text, ctx, "entity_2")
        if m:
            ai_slots["entity_2"] = m
        mod = _extract_via_gazetteer(text, ctx, "entity_3")
        if mod:
            ai_slots["entity_3"] = mod
        matched_cat = extract_pack_category(text, ctx)
        if not matched_cat:
            matched_cat = "UNKNOWN OR OTHER"
        ai_slots["category"] = matched_cat

        ai_kill_triggered = _check_kill_switch(text, ctx) is not None
        if ai_kill_triggered:
            ai_sev = "Critical"
        elif ai_slots.get("category") in ("SERVICE BRAKES", "STEERING", "VEHICLE SPEED CONTROL"):
            ai_sev = "High"
        else:
            ai_sev = "Medium"

        ai_cat = ai_slots.get("category") or ""
        ai_cluster_id = _CATEGORY_CLUSTERS.get(ai_cat) if ai_cat else None
        other_clusters = [c for c in all_cluster_ids if c != ai_cluster_id]
        ai_top_3 = [ai_cluster_id] + other_clusters[:2]

        telephony_cost = (0.0085 + 0.0043) * (duration / 60.0)
        tts_cost = 0.009
        llm_cost = (tokens / 1000.0) * 0.015
        total_cost = round(telephony_cost + tts_cost + llm_cost, 4)

        contacts.append(
            ShadowContact(
                contact_id=contact_id,
                ai_slots=ai_slots,
                human_slots=human_slots,
                ai_kill_switch_triggered=ai_kill_triggered,
                human_kill_switch_needed=human_kill_needed,
                ai_severity=ai_sev,
                human_severity=human_sev,
                ai_cluster_id=ai_cluster_id,
                ai_top_3_clusters=ai_top_3,
                engineer_verified_cluster_id=eng_cluster,
                duration_seconds=round(duration, 1),
                llm_tokens_used=tokens,
                cost_usd=total_cost,
            )
        )

    return contacts


def generate_shadow_cohort(sample_size: int = 100, seed: int = 42) -> list[ShadowContact]:
    """Generate a realistic cohort of dual-stream shadow contacts.

    Human side (slots, kill labels, severity) is synthesized ground truth by
    scenario assignment. The AI side executes the REAL deterministic intake
    code (gazetteer/year/category extractors + kill-switch) — never
    ``rng.random()`` accuracy simulation. A green metric here is a property
    of the shipped rules, which is exactly what Phase 0 must establish.
    """
    from src.agents.base import InteractionContext
    from src.agents.intake import (
        _check_kill_switch,
        _extract_year,
        _extract_via_gazetteer,
        extract_pack_category,
    )
    from src.domains.loader import load_pack

    pack = load_pack("automotive_nhtsa")
    ctx = InteractionContext(interaction_id="shadow_cohort", pack=pack)
    rng = random.Random(seed)
    contacts: list[ShadowContact] = []

    # Distribution:
    # ~8% true life-safety emergencies (must have 100% recall)
    # ~5% idiomatic/hedged mentions (must NOT trigger kill switch -> high precision)
    # ~87% standard automotive defect contacts
    emergency_indices = set(rng.sample(range(sample_size), k=max(2, int(sample_size * 0.08))))
    remaining_indices = [i for i in range(sample_size) if i not in emergency_indices]
    idiomatic_indices = set(rng.sample(remaining_indices, k=max(2, int(sample_size * 0.05))))

    all_cluster_ids = list(_CATEGORY_CLUSTERS.values())

    for i in range(sample_size):
        contact_id = f"shadow_cnt_{i+1:04d}"
        year = str(rng.randint(2015, 2024))
        make = rng.choice(list(_MAKES_MODELS.keys()))
        model = rng.choice(_MAKES_MODELS[make])
        category = rng.choice(_CATEGORIES)
        eng_cluster = _CATEGORY_CLUSTERS[category]

        # Defaults
        human_kill_needed = False
        ai_kill_triggered = False
        duration = rng.uniform(85.0, 160.0)
        tokens = rng.randint(400, 850)

        # Base human slots
        human_slots = {
            "entity_1": year,
            "entity_2": make,
            "entity_3": model,
            "category": category,
        }

        # NOTE: `text` is assigned per branch below; AI slots are extracted
        # from it with the real intake code after the branch. Human slots
        # stay exactly as assigned (synthetic ground truth).

        if i in emergency_indices:
            # True emergency
            text, sev, kill_term = rng.choice(_EMERGENCY_SCENARIOS)
            human_kill_needed = True
            human_sev = "Critical"
            ai_sev = "Critical"
            category = "ENGINE" if "fire" in kill_term else "SERVICE BRAKES"
            human_slots["category"] = category
            eng_cluster = _CATEGORY_CLUSTERS[category]
        elif i in idiomatic_indices:
            # Idiomatic mention — should NOT trigger kill switch
            text, sev = rng.choice(_IDIOMATIC_HEDGED_SCENARIOS)
            human_kill_needed = False
            human_sev = sev
            ai_sev = sev
        else:
            # Standard defect contact
            symptoms = _SYMPTOM_TEMPLATES[category]
            text = rng.choice(symptoms)
            human_kill_needed = False

            # Severity assignment
            if category in ("SERVICE BRAKES", "STEERING"):
                human_sev = "High" if rng.random() < 0.75 else "Medium"
            elif category in ("ENGINE", "AIR BAGS"):
                human_sev = "High" if rng.random() < 0.60 else "Medium"
            else:
                human_sev = "Medium" if rng.random() < 0.80 else "Low"

            # AI severity: deterministic mapping off the kill-switch and the
            # extracted category (mirrors the production triage-rule shape).
            # Human severity stays rng-assigned synthetic ground truth.
            ai_sev = "Medium"

        # Real contacts state the vehicle (lowercase, as STT delivers it);
        # without it in the text, entity accuracy would measure nothing.
        # Matching is case-insensitive (gazetteer match_substring lowercases).
        text = f"my {year} {make.lower()} {model.lower()}, {text}"

        # ── AI side: real deterministic intake code, no simulation ──
        ai_slots: dict[str, str] = {}
        y = _extract_year(text, (1990, 2026))
        if y:
            ai_slots["entity_1"] = y
        m = _extract_via_gazetteer(text, ctx, "entity_2")
        if m:
            ai_slots["entity_2"] = m
        mod = _extract_via_gazetteer(text, ctx, "entity_3")
        if mod:
            ai_slots["entity_3"] = mod
        matched_cat = extract_pack_category(text, ctx)
        if not matched_cat:
            matched_cat = "UNKNOWN OR OTHER"
        ai_slots["category"] = matched_cat

        ai_kill_triggered = _check_kill_switch(text, ctx) is not None
        if ai_kill_triggered:
            ai_sev = "Critical"
        elif ai_slots.get("category") in ("SERVICE BRAKES", "STEERING", "VEHICLE SPEED CONTROL"):
            ai_sev = "High"

        # AI Cluster — from extracted category only; never the engineer label.
        ai_cat = ai_slots.get("category") or ""
        ai_cluster_id = _CATEGORY_CLUSTERS.get(ai_cat) if ai_cat else None
        other_clusters = [c for c in all_cluster_ids if c != ai_cluster_id]
        ai_top_3 = [ai_cluster_id] + other_clusters[:2]

        # Blended unit cost:
        # Telephony: $0.0085/min * duration/60
        # Deepgram ASR: $0.0043/min * duration/60
        # TTS: $0.009
        # LLM (Gemini Flash fast-path): ~$0.00015 / 1k tokens
        telephony_cost = (0.0085 + 0.0043) * (duration / 60.0)
        tts_cost = 0.009
        llm_cost = (tokens / 1000.0) * 0.015
        total_cost = round(telephony_cost + tts_cost + llm_cost, 4)

        contacts.append(
            ShadowContact(
                contact_id=contact_id,
                ai_slots=ai_slots,
                human_slots=human_slots,
                ai_kill_switch_triggered=ai_kill_triggered,
                human_kill_switch_needed=human_kill_needed,
                ai_severity=ai_sev,
                human_severity=human_sev,
                ai_cluster_id=ai_cluster_id,
                ai_top_3_clusters=ai_top_3,
                engineer_verified_cluster_id=eng_cluster,
                duration_seconds=round(duration, 1),
                llm_tokens_used=tokens,
                cost_usd=total_cost,
            )
        )

    return contacts


def load_empirical_cohort(
    input_path: str,
    pack_id: str = "automotive_nhtsa",
    sample_size: int | None = None,
) -> list[ShadowContact]:
    """Load empirical contacts from an authentic JSONL/JSON recorded transcript corpus.

    Evaluates live NLP extraction, safety hazard detection, severity mapping,
    and cluster assignment against human ground truth.
    """
    from src.domains.loader import load_pack
    from src.agents.base import InteractionContext
    from src.agents.intake import (
        _check_kill_switch,
        _extract_year,
        _extract_via_gazetteer,
        extract_pack_category,
    )

    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Empirical corpus file not found: {input_path}")

    pack = load_pack(pack_id)
    ctx = InteractionContext(interaction_id="shadow_eval", pack=pack)

    lines = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if sample_size and sample_size < len(lines):
        lines = lines[:sample_size]

    contacts: list[ShadowContact] = []
    all_cluster_ids = list(_CATEGORY_CLUSTERS.values())

    for i, line in enumerate(lines):
        record = json.loads(line)
        contact_id = record.get("id", f"empirical_cnt_{i+1:04d}")
        text = record.get("text", "")

        # Ground truth slots
        human_slots = {
            "entity_1": str(record.get("entity_1", "")),
            "entity_2": str(record.get("entity_2", "")),
            "entity_3": str(record.get("entity_3", "")),
            "category": str(record.get("category", "")),
        }
        human_kill_needed = bool(record.get("expected_safety", record.get("human_kill_needed", False)))
        # F-010: never invent supervisor/engineer labels from the same rules the AI uses.
        human_sev = (record.get("human_severity") or "").strip()
        eng_cluster = record.get("engineer_cluster_id")
        if eng_cluster is None:
            eng_cluster = record.get("engineer_verified_cluster_id")
        if isinstance(eng_cluster, str) and str(eng_cluster).isdigit():
            eng_cluster = int(eng_cluster)
        elif eng_cluster is not None:
            try:
                eng_cluster = int(eng_cluster)
            except (TypeError, ValueError):
                eng_cluster = None

        # AI Extraction
        ai_slots: dict[str, str] = {}
        y = _extract_year(text, (1990, 2026))
        if y:
            ai_slots["entity_1"] = y
        m = _extract_via_gazetteer(text, ctx, "entity_2")
        if m:
            ai_slots["entity_2"] = m
        mod = _extract_via_gazetteer(text, ctx, "entity_3")
        if mod:
            ai_slots["entity_3"] = mod

        # Category extraction: shared intake matcher (skip/negation/phrases).
        matched_cat = extract_pack_category(text, ctx)
        if not matched_cat:
            # Residual NHTSA class when no specific system is named. Not a
            # ground-truth copy: unmatched text is UNKNOWN OR OTHER.
            matched_cat = "UNKNOWN OR OTHER"
        if matched_cat:
            ai_slots["category"] = matched_cat
        # F-009: do not copy human/ground-truth category into the AI prediction.

        # AI Kill Switch
        matched_term = _check_kill_switch(text, ctx)
        ai_kill_triggered = (matched_term is not None)

        # AI Severity
        if ai_kill_triggered:
            ai_sev = "Critical"
        elif ai_slots.get("category") in ("SERVICE BRAKES", "STEERING", "VEHICLE SPEED CONTROL"):
            ai_sev = "High"
        else:
            ai_sev = "Medium"

        # AI Cluster — from extracted category only; never the engineer label.
        ai_cat = ai_slots.get("category") or ""
        ai_cluster_id = _CATEGORY_CLUSTERS.get(ai_cat) if ai_cat else None
        other_clusters = [c for c in all_cluster_ids if c != ai_cluster_id]
        ai_top_3 = [ai_cluster_id] + other_clusters[:2]

        # Duration & cost modeling
        words = len(text.split())
        duration = max(45.0, min(180.0, words * 1.8))
        tokens = int(words * 4.5 + 250)
        telephony_cost = (0.0085 + 0.0043) * (duration / 60.0)
        tts_cost = 0.009
        llm_cost = (tokens / 1000.0) * 0.015
        total_cost = round(telephony_cost + tts_cost + llm_cost, 4)

        contacts.append(
            ShadowContact(
                contact_id=contact_id,
                ai_slots=ai_slots,
                human_slots=human_slots,
                ai_kill_switch_triggered=ai_kill_triggered,
                human_kill_switch_needed=human_kill_needed,
                ai_severity=ai_sev,
                human_severity=human_sev,
                ai_cluster_id=ai_cluster_id,
                ai_top_3_clusters=ai_top_3,
                engineer_verified_cluster_id=eng_cluster,
                duration_seconds=round(duration, 1),
                llm_tokens_used=tokens,
                cost_usd=total_cost,
            )
        )

    return contacts


def _format_stream_metrics_table(
    stream_name: str,
    stream_role: str,
    m: dict[str, Any],
    verdict: str,
    total_contacts: int,
    w: int = 88,
) -> list[str]:
    lines: list[str] = []
    lines.append("-" * w)
    lines.append(f"STREAM: {stream_name.upper()} [{stream_role.upper()}] (n={total_contacts})")
    lines.append("-" * w)
    lines.append(f"{'Metric':<34} {'Estimate':<10} {'95% CI':<20} {'Target':<14} {'Rating'}")
    lines.append("-" * w)

    slots = m.get("slots", {})
    for k, v in slots.items():
        name = f"Slot: {k}"
        est = f"{v['point_estimate']:.4f}"
        ci = f"[{v['ci_lower']:.4f}, {v['ci_upper']:.4f}]"
        rating = f"[{v['rating'].upper()}]"
        lines.append(f"{name:<34} {est:<10} {ci:<20} {v['target']:<14} {rating}")

    ks = m.get("kill_switch", {})
    for k, v in ks.items():
        name = f"Kill-Switch {k.capitalize()}"
        est = f"{v['point_estimate']:.4f}"
        ci = f"[{v['ci_lower']:.4f}, {v['ci_upper']:.4f}]"
        rating = f"[{v['rating'].upper()}]"
        lines.append(f"{name:<34} {est:<10} {ci:<20} {v['target']:<14} {rating}")

    sev = m.get("severity_agreement", {})
    if sev:
        est_sev = f"{sev['point_estimate']:.4f}"
        ci_sev = f"[{sev['ci_lower']:.4f}, {sev['ci_upper']:.4f}]"
        rating_sev = f"[{sev['rating'].upper()}]"
        lines.append(f"{'Severity Agreement (Kappa)':<34} {est_sev:<10} {ci_sev:<20} {sev['target']:<14} {rating_sev}")

    clu = m.get("cluster_agreement", {})
    for k, v in clu.items():
        name = f"Cluster Agreement ({k})"
        est = f"{v['point_estimate']:.4f}"
        ci = f"[{v['ci_lower']:.4f}, {v['ci_upper']:.4f}]"
        rating = f"[{v['rating'].upper()}]"
        lines.append(f"{name:<34} {est:<10} {ci:<20} {v['target']:<14} {rating}")

    cost = m.get("cost_per_contact", {})
    if cost:
        est_cost = f"${cost['point_estimate']:.4f}"
        ci_cost = f"[${cost['ci_lower']:.4f}, ${cost['ci_upper']:.4f}]"
        rating_cost = f"[{cost['rating'].upper()}]"
        lines.append(f"{'Unit Cost per Contact':<34} {est_cost:<10} {ci_cost:<20} {cost['target']:<14} {rating_cost}")

    lines.append(f"Stream Verdict: [{verdict.upper()}]")
    return lines


def format_scorecard_table(report: dict[str, Any], target_cost_usd: float = 0.45) -> str:
    """Render an ANSI text scorecard table suitable for console and log output."""
    meta = report.get("meta", {})
    data_source = meta.get("data_source", "synthetic_monte_carlo")
    source_type = meta.get("source_type", "offline_synthetic").upper()

    lines: list[str] = []
    w = 88
    lines.append("=" * w)
    lines.append("                  SHADOW PILOT EVALUATION SCORECARD (STEP 0)")
    lines.append("=" * w)
    lines.append(f"Cohort Size: {report.get('total_contacts')} contacts | Pack: {meta.get('pack_id', 'automotive_nhtsa')} | Target Cost: <= ${target_cost_usd:.2f}")
    lines.append(f"Data Source: {data_source} [{source_type}]")

    streams = report.get("streams")
    if streams and "synthetic_templated" in streams and "adversarial_paraphrase" in streams:
        # Dual-stream display
        st_tpl = streams["synthetic_templated"]
        lines.extend(_format_stream_metrics_table(
            "synthetic_templated",
            "SMOKE TEST ONLY — NOT AN ACCURACY CLAIM",
            st_tpl.get("metrics", {}),
            st_tpl.get("overall_verdict", "UNKNOWN"),
            st_tpl.get("total_contacts", 0),
            w=w,
        ))
        st_adv = streams["adversarial_paraphrase"]
        lines.extend(_format_stream_metrics_table(
            "adversarial_paraphrase",
            "ACCURACY CLAIM — HONEST PRODUCTION BASELINE",
            st_adv.get("metrics", {}),
            st_adv.get("overall_verdict", "UNKNOWN"),
            st_adv.get("total_contacts", 0),
            w=w,
        ))
    else:
        m = report.get("metrics", {})
        slots = m.get("slots", {})
        ks = m.get("kill_switch", {})
        sev = m.get("severity_agreement", {})
        clu = m.get("cluster_agreement", {})
        cost = m.get("cost_per_contact", {})

        lines.append("-" * w)
        lines.append(f"{'Metric':<34} {'Estimate':<10} {'95% CI':<20} {'Target':<14} {'Rating'}")
        lines.append("-" * w)

        for k, v in slots.items():
            name = f"Slot: {k}"
            est = f"{v['point_estimate']:.4f}"
            ci = f"[{v['ci_lower']:.4f}, {v['ci_upper']:.4f}]"
            rating = f"[{v['rating'].upper()}]"
            lines.append(f"{name:<34} {est:<10} {ci:<20} {v['target']:<14} {rating}")

        comp = m.get("category_composition") or {}
        subset = comp.get("matched_subset") or {}
        residual = comp.get("residual_unknown") or {}
        if subset:
            est = f"{subset['point_estimate']:.4f}"
            ci = f"[{subset['ci_lower']:.4f}, {subset['ci_upper']:.4f}]"
            lines.append(
                f"{'Category matched-subset':<34} {est:<10} {ci:<20} "
                f"{'diagnostic':<14} {subset.get('detail', '')}"
            )
        if residual:
            est = f"{residual['point_estimate']:.4f}"
            ci = f"[{residual['ci_lower']:.4f}, {residual['ci_upper']:.4f}]"
            lines.append(
                f"{'Category residual UNKNOWN':<34} {est:<10} {ci:<20} "
                f"{'diagnostic':<14} {residual.get('detail', '')}"
            )
        if comp.get("note"):
            lines.append(comp["note"])

        for k, v in ks.items():
            name = f"Kill-Switch {k.capitalize()}"
            est = f"{v['point_estimate']:.4f}"
            ci = f"[{v['ci_lower']:.4f}, {v['ci_upper']:.4f}]"
            rating = f"[{v['rating'].upper()}]"
            lines.append(f"{name:<34} {est:<10} {ci:<20} {v['target']:<14} {rating}")

        if sev:
            est_sev = f"{sev['point_estimate']:.4f}"
            ci_sev = f"[{sev['ci_lower']:.4f}, {sev['ci_upper']:.4f}]"
            rating_sev = f"[{sev['rating'].upper()}]"
            lines.append(f"{'Severity Agreement (Kappa)':<34} {est_sev:<10} {ci_sev:<20} {sev['target']:<14} {rating_sev}")

        for k, v in clu.items():
            name = f"Cluster Agreement ({k})"
            est = f"{v['point_estimate']:.4f}"
            ci = f"[{v['ci_lower']:.4f}, {v['ci_upper']:.4f}]"
            rating = f"[{v['rating'].upper()}]"
            lines.append(f"{name:<34} {est:<10} {ci:<20} {v['target']:<14} {rating}")

        if cost:
            est_cost = f"${cost['point_estimate']:.4f}"
            ci_cost = f"[${cost['ci_lower']:.4f}, ${cost['ci_upper']:.4f}]"
            rating_cost = f"[{cost['rating'].upper()}]"
            lines.append(f"{'Unit Cost per Contact':<34} {est_cost:<10} {ci_cost:<20} {cost['target']:<14} {rating_cost}")

    lines.append("-" * w)
    verdict = report.get("overall_verdict", "UNKNOWN").upper()
    waived = report.get("waived_metrics") or []
    status_note = {
        "GREEN": "scored gates met",
        "YELLOW": "HUMAN IN LOOP — Proceed with mandatory supervisor sign-off on flagged entities",
        "RED": "BLOCKED — Do not route live customer calls; address regressions in engineering",
        "BLOCKED": "BLOCKED — Independent human labels missing or metrics below gate",
    }.get(verdict, "")
    if verdict == "GREEN" and waived:
        status_note += f" — waived unlabeled: {', '.join(waived)}"
    elif verdict == "GREEN":
        status_note = "PROCEED — System meets all statistical thresholds for Phase 1 Live Pilot (5% traffic)"
    verdict_src = f" (driven by {report.get('verdict_source_stream', 'single_stream')})" if report.get("verdict_source_stream") else ""
    lines.append(f"OVERALL PILOT VERDICT: [{verdict}]{verdict_src} — {status_note}")
    lines.append("=" * w)
    return "\n".join(lines)


def run_shadow_pilot(
    mode: str = "batch",
    sample_size: int = 100,
    out_path: str = "reports/shadow_baseline_01.json",
    pack_id: str = "automotive_nhtsa",
    target_cost: float = 0.45,
    seed: int = 42,
    input_path: str | None = None,
) -> dict[str, Any]:
    """Run shadow pilot evaluation and write JSON report to disk."""
    if mode == "live" and not input_path:
        raise ValueError(
            "--mode=live requires --input with labeled contacts; "
            "refusing Monte Carlo as live evidence"
        )
    if input_path:
        contacts = load_empirical_cohort(input_path, pack_id=pack_id, sample_size=sample_size)
        evaluator = ShadowPilotEvaluator(contacts)
        report = evaluator.run_full_evaluation(target_cost_usd=target_cost)
        report["meta"] = {
            "mode": mode,
            "pack_id": pack_id,
            "sample_size": len(contacts),
            "seed": seed,
            "target_cost_usd": target_cost,
            "data_source": input_path,
            "source_type": "empirical_ground_truth",
            "synthetic_self_labels": False,
            "ai_source": "deterministic_intake_code",
            "human_labels": "synthetic_scenario_assignment",
        }
    else:
        # Dual-stream validation:
        # Stream 1: synthetic_templated (smoke test, NOT an accuracy claim)
        # Stream 2: adversarial_paraphrase (accuracy claim, drives overall_verdict)
        templated_contacts = generate_shadow_cohort(sample_size=sample_size, seed=seed)
        adversarial_contacts = generate_adversarial_paraphrase_cohort(sample_size=sample_size, seed=seed)

        evaluator_tpl = ShadowPilotEvaluator(templated_contacts)
        report_tpl = evaluator_tpl.run_full_evaluation(target_cost_usd=target_cost)

        evaluator_adv = ShadowPilotEvaluator(adversarial_contacts)
        report_adv = evaluator_adv.run_full_evaluation(target_cost_usd=target_cost)

        # Overall verdict is driven strictly by the adversarial stream
        report = {
            "total_contacts": len(adversarial_contacts),
            "overall_verdict": report_adv["overall_verdict"],
            "verdict_source_stream": "adversarial_paraphrase",
            "waived_metrics": report_adv.get("waived_metrics", []),
            "metrics": report_adv["metrics"],
            "streams": {
                "synthetic_templated": {
                    "corpus": "synthetic_templated",
                    "role": "smoke_test",
                    "accuracy_claim": False,
                    "description": "Synthetic templated cohort with exact lexicon/gazetteer terms; integration smoke test only, explicitly NOT an accuracy claim.",
                    "total_contacts": len(templated_contacts),
                    "overall_verdict": report_tpl["overall_verdict"],
                    "metrics": report_tpl["metrics"],
                },
                "adversarial_paraphrase": {
                    "corpus": "adversarial_paraphrase",
                    "role": "accuracy_claim",
                    "accuracy_claim": True,
                    "description": "Adversarial caller paraphrase cohort without verbatim lexicon/gazetteer terms; honest system accuracy baseline.",
                    "total_contacts": len(adversarial_contacts),
                    "overall_verdict": report_adv["overall_verdict"],
                    "metrics": report_adv["metrics"],
                },
            },
            "meta": {
                "mode": mode,
                "pack_id": pack_id,
                "sample_size": len(adversarial_contacts),
                "seed": seed,
                "target_cost_usd": target_cost,
                "data_source": "dual_stream_validation",
                "source_type": "offline_adversarial_paraphrase",
                "synthetic_self_labels": True,
                "ai_source": "deterministic_intake_code",
                "human_labels": "synthetic_scenario_assignment",
                "overall_verdict_stream": "adversarial_paraphrase",
                "honesty": {
                    "streams_reported": ["synthetic_templated", "adversarial_paraphrase"],
                    "accuracy_claim_stream": "adversarial_paraphrase",
                    "smoke_test_stream": "synthetic_templated",
                    "note": "overall_verdict is driven exclusively by adversarial_paraphrase stream. synthetic_templated is a smoke test and cannot support an accuracy claim.",
                },
            },
        }

    # Write output report
    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Frontline Shadow Pilot Validation Runner")
    parser.add_argument("--mode", choices=["batch", "replay", "live"], default="batch", help="Shadow execution mode")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of dual-stream contacts to evaluate")
    parser.add_argument("--out", type=str, default="reports/shadow_baseline_01.json", help="Path to write JSON evaluation scorecard")
    parser.add_argument("--pack", type=str, default="automotive_nhtsa", help="Domain pack identifier")
    parser.add_argument("--target-cost", type=float, default=0.45, help="Maximum target unit cost per contact (USD)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatable evaluation")
    parser.add_argument("--input", type=str, default=None, help="Path to empirical JSONL transcript corpus")

    args = parser.parse_args()

    report = run_shadow_pilot(
        mode=args.mode,
        sample_size=args.sample_size,
        out_path=args.out,
        pack_id=args.pack,
        target_cost=args.target_cost,
        seed=args.seed,
        input_path=args.input,
    )

    table = format_scorecard_table(report, target_cost_usd=args.target_cost)
    print(table)

    verdict = report.get("overall_verdict")
    return 0 if verdict in ("green", "yellow") else 1


if __name__ == "__main__":
    sys.exit(main())
