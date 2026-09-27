"""The gazetteer alias column is opt-in via the CSV header.

Two real column-2 meanings exist in this repo:

  finance_cfpb/gazetteers/products.csv    value,alias      <- column 2 IS an alias
  automotive_nhtsa/gazetteers/*.csv       value,frequency  <- column 2 is a COUNT

`Gazetteer.from_csv` used to treat column 2 as an alias unconditionally, so
every NHTSA frequency became a lookup key: `lookup("1610") -> "AIR BAGS"`.
That reaches live slot extraction through `src/agents/intake.py`
`_extract_via_gazetteer`, so a caller who said a four-digit number got a
vehicle system assigned from it.

`src/ml_runtime/category_zeroshot.py` already carries a downstream
`not alias.isdigit()` workaround for this; these tests pin the root cause.
"""

from __future__ import annotations

from src.domains.loader import Gazetteer, load_pack


def _write(tmp_path, name: str, body: str):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return Gazetteer.from_csv(p)


def test_frequency_column_is_not_an_alias(tmp_path):
    """A `value,frequency` header means column 2 is a count, not a lookup key."""
    gaz = _write(
        tmp_path,
        "categories.csv",
        "value,frequency\nAIR BAGS,1610\nSERVICE BRAKES,3120\n",
    )

    assert gaz.values == ("AIR BAGS", "SERVICE BRAKES")
    # The frequency must not be resolvable as a value.
    assert gaz.lookup("1610") is None
    assert gaz.lookup("3120") is None
    # ...nor pollute the membership set used for slot validation.
    assert gaz.value_set == {"air bags", "service brakes"}
    # ...nor be extractable from a caller's utterance.
    assert gaz.match_substring("my 2019 bill was 1610 dollars") is None
    # The real values still work.
    assert gaz.lookup("air bags") == "AIR BAGS"
    assert gaz.match_substring("the air bags did not deploy") == "AIR BAGS"


def test_alias_column_still_works(tmp_path):
    """A `value,alias` header keeps the aliasing behaviour finance_cfpb relies on."""
    gaz = _write(
        tmp_path,
        "companies.csv",
        "value,alias\nJPMORGAN CHASE & CO.,Chase\nBANK OF AMERICA,BofA\n",
    )

    assert gaz.lookup("chase") == "JPMORGAN CHASE & CO."
    assert gaz.lookup("bofa") == "BANK OF AMERICA"
    assert gaz.lookup("jpmorgan chase & co.") == "JPMORGAN CHASE & CO."
    assert "chase" in gaz.value_set
    # An alias is not a separate canonical value.
    assert gaz.values == ("JPMORGAN CHASE & CO.", "BANK OF AMERICA")


def test_single_column_gazetteer_unaffected(tmp_path):
    """A bare `value` header (most packs) keeps working."""
    gaz = _write(tmp_path, "issues.csv", "value\nBILLING DISPUTE\nFRAUD\n")

    assert gaz.values == ("BILLING DISPUTE", "FRAUD")
    assert gaz.value_set == {"billing dispute", "fraud"}


def test_unknown_second_column_is_ignored(tmp_path):
    """An unrecognised column-2 header fails safe: ignored, never aliased.

    Degrading to "no alias" loses a matching opportunity; treating an unknown
    column as an alias corrupts the controlled vocabulary. Prefer the former.
    """
    gaz = _write(tmp_path, "makes.csv", "value,notes\nHONDA,imported 1959\n")

    assert gaz.lookup("honda") == "HONDA"
    assert gaz.lookup("imported 1959") is None
    assert gaz.value_set == {"honda"}


def test_real_nhtsa_category_gazetteer_has_no_numeric_keys():
    """End-to-end: the live pack's category vocabulary is values only.

    This is the invariant Phase 0 depends on — `records.category` and the
    live-call `category` slot must be drawn from the same controlled set.
    """
    pack = load_pack("automotive_nhtsa")
    gaz = pack.gazetteer_for_slot("category")
    assert gaz is not None

    numeric = sorted(k for k in gaz.value_set if k.isdigit())
    assert numeric == [], f"frequencies leaked into the category vocabulary: {numeric}"

    # One membership key per declared value, nothing more.
    assert len(gaz.value_set) == len(gaz.values)
    assert gaz.value_set == {v.lower() for v in gaz.values}
