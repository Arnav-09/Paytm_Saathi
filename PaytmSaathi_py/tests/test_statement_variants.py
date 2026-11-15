"""Layout-tolerant statement parsing: the same transactions exported in 12 different bank / card layouts
(samples/statements/variants/, made by samples/make_statement_variants.py) must all read back the same.

expected.json is produced by the generator from its source data, so these tests check the parser against the truth,
not against its own earlier output.
"""

import json
import runpy
from datetime import date
from pathlib import Path

import pytest

from ai_engine import money, statements
from ai_engine.exceptions import InvalidFileError

VARIANTS = Path(__file__).resolve().parent.parent / "samples" / "statements" / "variants"
GENERATOR = VARIANTS.parent.parent / "make_statement_variants.py"
if not (VARIANTS / "expected.json").exists():
    runpy.run_path(str(GENERATOR), run_name="__main__")
EXPECTED = json.loads((VARIANTS / "expected.json").read_text(encoding="utf-8"))
FILES = EXPECTED["files"]


@pytest.fixture(scope="module")
def parsed():
    return {name: statements.parse_statement(name, (VARIANTS / name).read_bytes()) for name in FILES}


@pytest.mark.parametrize("name", list(FILES))
def test_variant_counts_and_totals(parsed, name):
    s, e = parsed[name], FILES[name]
    assert s.kind == e["kind"]
    assert len(s.transactions) == e["count"]
    assert sum(t.amount for t in s.transactions if t.type == "debit") == pytest.approx(e["total_debit"])
    assert sum(t.amount for t in s.transactions if t.type == "credit") == pytest.approx(e["total_credit"])
    for field in ("closing_balance", "opening_balance", "card_total_due", "card_minimum_due", "card_due_date",
                  "card_credit_limit", "period_start", "period_end"):
        if field in e:
            assert getattr(s, field) == pytest.approx(e[field]) if isinstance(e[field], float) else getattr(s, field) == e[field], field
    assert bool(s.warnings) == bool(e.get("warning"))


@pytest.mark.parametrize("name", list(FILES))
def test_variant_reads_exactly_the_reference_transactions(parsed, name):
    """Same dates, amounts and directions as the source data, whatever the layout."""
    ref = sorted(tuple(r) for r in EXPECTED["reference"][FILES[name]["same_as"]])
    got = sorted((t.date, t.amount, t.type) for t in parsed[name].transactions)
    assert got == ref


@pytest.mark.parametrize("name", list(FILES))
def test_variant_spot_checks(parsed, name):
    for check in FILES[name]["checks"]:
        hits = [t for t in parsed[name].transactions if check["contains"].lower() in t.description.lower()]
        assert hits, f"{name}: no transaction mentions {check['contains']!r}"
        t = hits[0]
        for key, field in (("type", "type"), ("amount", "amount"), ("date", "date"), ("category", "category")):
            if key in check:
                assert getattr(t, field) == check[key], f"{name}: {check['contains']} {key}"
        if "source_has" in check:
            assert check["source_has"] in t.source


@pytest.mark.parametrize("name", list(FILES))
def test_every_variant_masks_long_numbers_and_keeps_a_source(parsed, name):
    for t in parsed[name].transactions:
        assert t.source.startswith(name) and ("row" in t.source or "page" in t.source)
        assert "403011554321" not in t.description  # account / reference numbers masked


def test_variants_give_the_same_money_report():
    """The analysis does not depend on which bank layout the user uploaded."""
    reports = [money.analyse(statements.parse_statement(n, (VARIANTS / n).read_bytes()), date(2026, 11, 15))
               for n in ("hdfc_style.csv", "bank_table.pdf", "excel_dates.xlsx", "html_export.xls")]
    totals = {tuple((c.category, round(c.total, 2)) for c in r.categories) for r in reports}
    assert len(totals) == 1


@pytest.mark.parametrize("pdf,plain", [("card_layout2.pdf", "card_export.csv"), ("bank_table.pdf", "no_header.csv"),
                                       ("html_export.xls", "no_header.csv"), ("drcr_column.tsv", "no_header.csv")])
def test_descriptions_match_across_layouts(parsed, pdf, plain):
    """Columns that are not the description (reward points, ref numbers, branch codes) must not leak into it."""
    def rows(name):
        return sorted((t.date, t.amount, t.description) for t in parsed[name].transactions)

    assert rows(pdf) == rows(plain)


def test_wrapped_pdf_description_is_joined(parsed):
    salary = next(t for t in parsed["bank_table.pdf"].transactions if "SALARY" in t.description)
    assert "ACME TECHNOLOGIES PVT LTD" in salary.description and salary.category == "Income"


def test_card_export_negative_means_payment(parsed):
    by_desc = {t.description: t.type for t in parsed["card_export.csv"].transactions}
    assert by_desc["BBPS PAYMENT 7788"] == "credit" and by_desc["CROMA RETAIL ELECTRONICS"] == "debit"


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("01/11/2026", date(2026, 11, 1)), ("01-11-26", date(2026, 11, 1)), ("01.11.2026", date(2026, 11, 1)),
    ("2026-11-01", date(2026, 11, 1)), ("2026-11-01 09:15:22", date(2026, 11, 1)), ("2026-11-01T09:15:22", date(2026, 11, 1)),
    ("1 Nov 2026", date(2026, 11, 1)), ("01-Nov-26", date(2026, 11, 1)), ("Nov 01, 2026", date(2026, 11, 1)),
    ("1st November 2026", date(2026, 11, 1)), ("Sept 3, 2026", date(2026, 9, 3)), ("11/25/2026", date(2026, 11, 25)),
    ("20261101", date(2026, 11, 1)), (46327, date(2026, 11, 1)), ("Opening Balance", None), ("", None),
    ("32/13/2026", None), ("12345.60", None),
])
def test_parse_date_styles(text, expected):
    assert statements.parse_date(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("1,234.50", 1234.5), ("₹1,234.50 Dr", -1234.5), ("Rs. 1,00,000.00", 100000.0), ("INR 500", 500.0),
    ("(500.00)", -500.0), ("-200", -200.0), ("200.00-", -200.0), ("−75.25", -75.25), ("+₹85,000.00", 85000.0),
    ("12,345.60 Cr", 12345.6), ("CR 99.00", 99.0), ("1,234.5 debit", -1234.5), ("-", None), ("abc", None),
    ("01/11/2026", None), ("1234,56", None),
])
def test_parse_money_styles(text, expected):
    assert statements.parse_money(text) == expected


@pytest.mark.parametrize("headers,expected", [
    (["Txn Date", "Value Date", "Narration", "Withdrawal Amt. (INR )", "Deposit Amt. (INR )", "Closing Balance"],
     {"date": 0, "desc": 2, "debit": 3, "credit": 4, "balance": 5}),
    (["Date & Time", "Details", "Amount (₹)", "Dr / Cr"], {"date": 0, "desc": 1, "amount": 2, "type": 3}),
    (["Posting Date", "Merchant Name", "Billing Amount", "Balance Amount"],
     {"date": 0, "desc": 1, "amount": 2, "balance": 3}),
])
def test_column_matching_by_meaning(headers, expected):
    assert statements._map_columns(headers) == expected


def test_old_binary_xls_gets_a_clear_message():
    with pytest.raises(InvalidFileError, match="save it as .xlsx or CSV"):
        statements.parse_statement("old.xls", b"\xd0\xcf\x11\xe0" + b"\x00" * 100)


def test_utf16_tab_export():
    text = "Date\tDescription\tDebit\tCredit\n01/11/2026\tSWIGGY\t450.00\t\n02/11/2026\tSALARY ACME\t\t50000.00\n"
    s = statements.parse_statement("export.txt", text.encode("utf-16"))
    assert [(t.type, t.amount) for t in s.transactions] == [("debit", 450.0), ("credit", 50000.0)]


def test_text_file_that_is_not_a_statement_is_refused():
    with pytest.raises(statements.StatementError):
        statements.parse_statement("letter.txt", b"Dear Applicant,\nWe are unable to approve your loan.\n")
