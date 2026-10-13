"""ISO 3779 Vehicle Identification Number (VIN) validator."""

from __future__ import annotations

import re

VIN_TRANSLITERATION_MAP = {
    'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8,
    'J': 1, 'K': 2, 'L': 3, 'M': 4, 'N': 5, 'P': 7, 'R': 9,
    'S': 2, 'T': 3, 'U': 4, 'V': 5, 'W': 6, 'X': 7, 'Y': 8, 'Z': 9,
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
}

VIN_WEIGHTS = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]


def validate_iso3779_vin(vin: str) -> bool:
    """
    Validates a 17-character VIN against the ISO 3779 check digit algorithm.
    Enforces exclusion of illegal characters: I, O, Q.
    """
    if not vin or not isinstance(vin, str):
        return False

    raw_vin = vin.strip().upper()
    # Check for illegal ISO characters explicitly
    if any(c in raw_vin for c in ('I', 'O', 'Q')):
        return False

    clean_vin = re.sub(r"[^A-HJ-NPR-Z0-9]", "", raw_vin)
    if len(clean_vin) != 17:
        return False

    computed_sum = 0
    for idx, char in enumerate(clean_vin):
        val = VIN_TRANSLITERATION_MAP.get(char, 0)
        computed_sum += val * VIN_WEIGHTS[idx]

    remainder = computed_sum % 11
    expected_check = "X" if remainder == 10 else str(remainder)
    return clean_vin[8] == expected_check
