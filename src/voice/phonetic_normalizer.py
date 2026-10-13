"""Phonetic alphabet and spoken alphanumeric normalizer."""

from __future__ import annotations

import re

PHONETIC_LEXICON: dict[str, str] = {
    "alpha": "A", "adam": "A",
    "bravo": "B", "boy": "B",
    "charlie": "C", "charles": "C",
    "delta": "D", "david": "D",
    "echo": "E", "edward": "E",
    "foxtrot": "F", "frank": "F",
    "golf": "G", "george": "G",
    "hotel": "H", "henry": "H",
    "india": "I", "ida": "I",
    "juliet": "J", "john": "J",
    "kilo": "K", "king": "K",
    "lima": "L", "lincoln": "L",
    "mike": "M", "mary": "M",
    "november": "N", "nancy": "N",
    "oscar": "O", "ocean": "O",
    "papa": "P", "paul": "P",
    "quebec": "Q", "queen": "Q",
    "romeo": "R", "robert": "R",
    "sierra": "S", "sam": "S",
    "tango": "T", "tom": "T",
    "uniform": "U", "union": "U",
    "victor": "V",
    "whiskey": "W", "william": "W",
    "x-ray": "X", "xray": "X",
    "yankee": "Y", "yellow": "Y",
    "zulu": "Z", "zebra": "Z",
    "zero": "0", "oh": "0",
    "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}


def parse_spoken_alphanumerics(transcript: str) -> str:
    """
    Converts phrases like: '1 F as in Frank T Y 2 8 Kilo' -> '1FTY28K'
    """
    text = (transcript or "").lower()

    def _as_in(match: re.Match[str]) -> str:
        spoken = match.group(1)
        word = match.group(2)
        # Known phonetic word wins. An unknown word must not drop the letter.
        if word in PHONETIC_LEXICON:
            return PHONETIC_LEXICON[word].lower()
        return spoken

    text = re.sub(r"([a-z0-9])\s+as\s+in\s+([a-z]+)", _as_in, text)
    # Keep the lexicon key intact. A hyphen split would turn "x-ray" into "x" + "ray".
    text = text.replace("x-ray", "xray")
    tokens = re.split(r"[\s\-]+", text)

    extracted_chars: list[str] = []
    for token in tokens:
        clean_token = re.sub(r'[^a-z0-9]', '', token)
        if not clean_token:
            continue
        if clean_token in PHONETIC_LEXICON:
            extracted_chars.append(PHONETIC_LEXICON[clean_token])
        elif len(clean_token) == 1:
            extracted_chars.append(clean_token.upper())

    return "".join(extracted_chars)
