"""Hindi / Hinglish detection and normalisation.  [AI-LAYER]

Detection is deterministic and offline. Normalisation to English uses Muse
Spark when a key is configured; otherwise (or on any failure) it falls back
to the existing glossary in ``src.frontline.i18n.normalize_to_english``,
which is left untouched.

Both the original and the normalised text are always returned so the caller
can store the pair.
"""

from __future__ import annotations

import re
from typing import Any

from src.frontline.i18n import detect_language, normalize_to_english

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_LATIN_WORD = re.compile(r"[a-z]+")

# Romanised Hindi function words / common vehicle-complaint vocabulary. A
# token set, not substrings, so English "ho" in "hold" never matches.
_HINGLISH_MARKERS = frozenset({
    "hai", "hain", "nahi", "nahin", "nhi", "mera", "meri", "mere", "mujhe",
    "gaadi", "gadi", "gaddi", "raha", "rahi", "rahe", "gaya", "gayi", "gaye",
    "kya", "kyun", "kyunki", "bahut", "bohot", "thoda", "accha", "acha",
    "theek", "thik", "wala", "wali", "sath", "saath", "aap", "aapka", "hum",
    "humara", "kab", "kaise", "abhi", "kal", "aaj", "jab", "tab", "phir",
    "lekin", "par", "aur", "bhi", "karta", "karti", "karna", "kar", "ho",
    "hota", "hoti", "lag", "lagta", "lagti", "chal", "chalti", "chalta",
    "band", "awaaz", "aawaz", "dikkat", "pareshani", "samasya", "ki", "ka",
    "ke", "se", "me", "mein", "pe", "ko", "tha", "thi", "the",
})
# Markers that are also common English words; they count only alongside a
# strong marker.
_WEAK = frozenset({"par", "band", "me", "the", "ho", "kar", "lag", "chal", "tab", "hum", "kal", "se", "pe", "ko", "ka", "ki", "ke"})


def detect(text: str) -> dict[str, Any]:
    """Return ``{"lang": "hi"|"hinglish"|<existing>, "hinglish_markers": [...]}``.

    ``hi`` = Devanagari script present (same rule as the existing detector).
    ``hinglish`` = Latin script with >= 2 distinct romanised-Hindi markers, at
    least one of them strong. Everything else defers to the existing
    ``src.frontline.i18n.detect_language``.
    """
    raw = text or ""
    if _DEVANAGARI.search(raw):
        return {"lang": "hi", "hinglish_markers": []}
    tokens = set(_LATIN_WORD.findall(raw.lower()))
    hits = sorted(tokens & _HINGLISH_MARKERS)
    strong = [h for h in hits if h not in _WEAK]
    if len(hits) >= 2 and strong:
        return {"lang": "hinglish", "hinglish_markers": hits}
    return {"lang": detect_language(raw), "hinglish_markers": []}


def deterministic_normalise(text: str, lang: str) -> dict[str, Any]:
    """Existing glossary path (unchanged); Hinglish passes through as-is."""
    if lang == "hinglish":
        return {"original": text or "", "normalized_en": text or "", "lang": "hinglish", "method": "identity"}
    return normalize_to_english(text or "", lang=lang)


def accept_ai_normalisation(original: str, normalized: str, confidence: float, min_conf: float) -> bool:
    """Cheap sanity checks on a model translation before it is trusted."""
    norm = (normalized or "").strip()
    if not norm or confidence < min_conf:
        return False
    if _DEVANAGARI.search(norm):
        return False  # not English
    if len(norm) > max(400, 4 * len(original or "")):
        return False  # model padded / invented content
    # Numbers (years, mileage) must survive translation unchanged: a number in
    # the English text that the caller never said is invented data.
    orig_nums = set(re.findall(r"\d+", _devanagari_digits_to_ascii(original or "")))
    for n in re.findall(r"\d+", norm):
        if n not in orig_nums:
            return False
    return True


_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def _devanagari_digits_to_ascii(text: str) -> str:
    return (text or "").translate(_DEV_DIGITS)


__all__ = ["accept_ai_normalisation", "deterministic_normalise", "detect"]
