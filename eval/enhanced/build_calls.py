"""Build the labelled SYNTHETIC call set for the enhanced-vs-deterministic eval.  [AI-LAYER]

Every row is synthetic (written from templates, no customer data) and is
marked ``"synthetic": true``. Labels are the values a correct intake should
produce under the automotive_nhtsa pack (makes/models/categories gazetteers).

    python -m eval.enhanced.build_calls   # rewrites eval/enhanced/calls.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).with_name("calls.jsonl")

VEHICLES = [
    ("HONDA", "होंडा", "CR-V", "2019"), ("TOYOTA", "टोयोटा", "CAMRY", "2018"), ("FORD", "फोर्ड", "ESCAPE", "2020"),
    ("HYUNDAI", "ह्युंडई", "ELANTRA", "2017"), ("KIA", "किआ", "SPORTAGE", "2021"), ("HONDA", "होंडा", "CIVIC", "2016"),
    ("TOYOTA", "टोयोटा", "COROLLA", "2022"), ("FORD", "फोर्ड", "EXPLORER", "2019"), ("HYUNDAI", "ह्युंडई", "TUCSON", "2020"),
    ("KIA", "किआ", "SORRENTO", "2018"),
]

# (category, kill_switch, english, hindi, hinglish)
SYMPTOMS = [
    ("SERVICE BRAKES", False, "the brakes make a loud grinding noise when I stop",
     "रुकते समय ब्रेक से तेज़ घिसने की आवाज़ आती है", "rukte waqt brakes se bahut grinding ki awaaz aati hai"),
    ("SERVICE BRAKES", True, "the brakes failed completely on the highway",
     "हाईवे पर ब्रेक पूरी तरह फेल हो गए", "highway pe brakes poori tarah fail ho gaye"),
    ("STEERING", False, "the steering wheel is very hard to turn at low speed",
     "धीमी रफ़्तार पर स्टीयरिंग घुमाना बहुत मुश्किल है", "slow speed pe steering ghumana bahut mushkil hai"),
    ("AIR BAGS", False, "the airbag warning light stays on all the time",
     "एयरबैग की चेतावनी लाइट हमेशा जलती रहती है", "airbag ki warning light hamesha on rehti hai"),
    ("ENGINE", False, "the engine stalls at traffic lights",
     "ट्रैफिक लाइट पर इंजन बंद हो जाता है", "traffic signal pe engine band ho jata hai"),
    ("FUEL SYSTEM", True, "smells of fuel and there was smoke coming from the hood",
     "पेट्रोल की गंध आती है और बोनट से धुआं निकला", "petrol ki smell aati hai aur bonnet se smoke nikla"),
    ("ELECTRICAL SYSTEM", False, "the battery drains overnight and the dashboard flickers",
     "बैटरी रात भर में खत्म हो जाती है और डैशबोर्ड टिमटिमाता है", "battery raat bhar mein drain ho jati hai aur dashboard flicker karta hai"),
    ("POWER TRAIN", False, "the transmission jerks when shifting gears",
     "गियर बदलते समय ट्रांसमिशन झटका देता है", "gear badalte waqt transmission jhatka deta hai"),
    ("SUSPENSION", False, "there is a clunking noise from the front suspension over bumps",
     "गड्ढों पर आगे के सस्पेंशन से खटखट की आवाज़ आती है", "speed breaker pe front suspension se khat khat ki awaaz aati hai"),
    ("TIRES", False, "the front tire tread is separating",
     "आगे के टायर की ट्रेड अलग हो रही है", "front tyre ki tread alag ho rahi hai"),
]


def build() -> list[dict]:
    rows: list[dict] = []
    for i, (cat, ks, en, hi, hl) in enumerate(SYMPTOMS):
        for j in range(2):
            make, make_hi, model, year = VEHICLES[(i + j * 5) % len(VEHICLES)]
            label = {"entity_2": make, "entity_3": model, "entity_1": year, "category": cat}
            rows.append({"id": f"en_{i:02d}_{j}", "lang": "en", "synthetic": True, "kill_switch": ks,
                         "text": f"My {year} {make.title()} {model} - {en}.", "expected": label})
            rows.append({"id": f"hi_{i:02d}_{j}", "lang": "hi", "synthetic": True, "kill_switch": ks,
                         "text": f"मेरी {year} की {make_hi} {model} में समस्या है, {hi}।", "expected": label})
            rows.append({"id": f"hl_{i:02d}_{j}", "lang": "hinglish", "synthetic": True, "kill_switch": ks,
                         "text": f"Meri {make.title()} {model} {year} mein problem hai, {hl}.", "expected": label})
    return rows


def main() -> int:
    rows = build()
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} synthetic calls to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
