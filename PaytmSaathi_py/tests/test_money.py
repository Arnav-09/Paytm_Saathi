"""Money Clarity: statement parsing (CSV/Excel/PDF), masking, categories, recurring and duplicate detection,
fees, month-over-month change, upcoming dues, cash-flow projection, action plan, evidence."""

import io
from datetime import date
from pathlib import Path

import pytest

from ai_engine import money, statements
from ai_engine.exceptions import InvalidFileError
from ai_engine.session_tools import SessionData, build_session_tools, compact_report

SAMPLES = Path(__file__).resolve().parent.parent / "samples" / "statements"
AS_OF = date(2026, 11, 15)


@pytest.fixture(scope="module", autouse=True)
def samples():
    if not (SAMPLES / "credit_card_nov_2026.pdf").exists():
        import runpy

        runpy.run_path(str(SAMPLES.parent / "make_statement_samples.py"), run_name="__main__")


def load(name):
    return statements.parse_statement(name, (SAMPLES / name).read_bytes())


def csv_bytes(rows, header="Date,Description,Amount,Type"):
    return (header + "\n" + "\n".join(rows)).encode()


# ----------------------------- parsing --------------------------------------
def test_bank_csv_and_excel_parse_the_same():
    a, b = load("bank_statement_oct_nov_2026.csv"), load("bank_statement_oct_nov_2026.xlsx")
    assert len(a.transactions) == len(b.transactions) == 30
    assert a.closing_balance == b.closing_balance == pytest.approx(44550.30)
    assert a.kind == "bank" and a.period_start == "2026-10-03" and a.period_end == "2026-11-13"
    assert a.transactions[0].source.endswith("row 6")  # preamble lines skipped, row numbers kept


def test_card_pdf_metadata_and_types():
    c = load("credit_card_nov_2026.pdf")
    assert c.kind == "card" and c.card_due_date == "2026-12-05"
    assert c.card_total_due == 38450 and c.card_minimum_due == 1930 and c.card_credit_limit == 100000
    assert (c.period_start, c.period_end) == ("2026-10-16", "2026-11-15")
    pay = next(t for t in c.transactions if "PAYMENT RECEIVED" in t.description)
    assert pay.type == "credit" and "page 1, line" in pay.source


def test_single_amount_column_with_type_and_signs():
    s = statements.parse_statement("x.csv", csv_bytes([
        "01/11/2026,SALARY ACME,50000,CR", "02/11/2026,SWIGGY ORDER,-450.50,", "03/11/2026,AMAZON,1200,DR"]))
    types = [(t.merchant, t.type, t.amount) for t in s.transactions]
    assert types == [("Salary Acme", "credit", 50000.0), ("Swiggy", "debit", 450.5), ("Amazon", "debit", 1200.0)]


def test_separate_debit_credit_columns_and_date_formats():
    s = statements.parse_statement("x.csv", csv_bytes(
        ["2026-11-01,NETFLIX.COM,649.00,", "05 Nov 2026,REFUND FLIPKART,,999.00"], header="Txn Date,Narration,Withdrawal Amt.,Deposit Amt."))
    assert [(t.date, t.type, t.category) for t in s.transactions] == [
        ("2026-11-01", "debit", "Subscriptions"), ("2026-11-05", "credit", "Refunds & cashback")]


def test_sensitive_numbers_are_masked_on_read():
    s = statements.parse_statement("x.csv", csv_bytes([
        "01/11/2026,CARD 4111 1111 1111 1111 PURCHASE AMAZON,500,DR",
        "02/11/2026,NEFT TO A/C 123456789012345 RENT,18000,DR"]))
    text = " ".join(t.description for t in s.transactions)
    assert "4111 1111" not in text and "123456789012345" not in text and "XXXX2345" in text


@pytest.mark.parametrize("name,data,err", [
    ("x.csv", b"", InvalidFileError), ("x.xlsx", b"not a zip", InvalidFileError), ("x.pdf", b"MZ fake", InvalidFileError),
    ("x.docx", b"PK\x03\x04", InvalidFileError), ("x.csv", b"hello,world\n1,2\n", statements.StatementError),
])
def test_bad_files_rejected(name, data, err):
    with pytest.raises(err):
        statements.parse_statement(name, data)


def test_categories_match_whole_words():
    cat = lambda d, t="debit": statements.categorise(d, t)[0]  # noqa: E731
    assert cat("LIC OF INDIA PREMIUM") == "Insurance"          # 'emi' inside 'premium' must not match EMI
    assert cat("NACH DR BAJAJ FINSERV EMI") == "EMI & loans"
    assert cat("FINANCE CHARGES") == "Fees & charges"          # plural
    assert cat("CURRENT ACCOUNT TRANSFER") != "Rent & housing"  # 'rent' inside 'current'
    assert cat("NACH NETFLIX") == "Subscriptions"
    assert cat("SALARY ACME", "credit") == "Income" and cat("SWIGGY", "credit") == "Money in"


# ----------------------------- analysis -------------------------------------
@pytest.fixture(scope="module")
def bank_report():
    return money.analyse(load("bank_statement_oct_nov_2026.csv"), AS_OF)


@pytest.fixture(scope="module")
def card_report():
    return money.analyse(statements.merge_statements([load("credit_card_oct_2026.pdf"), load("credit_card_nov_2026.pdf")]), AS_OF)


def test_recurring_detection(bank_report):
    rec = {r.merchant: r for r in bank_report.recurring}
    assert rec["Rent"].frequency == "monthly" and rec["Rent"].next_expected == "2026-12-03"
    assert rec["Bajaj Finserv"].next_expected == "2026-12-05" and rec["Spotify"].confidence == "medium"
    assert rec["Netflix"].confidence == "low"  # price changed, so only an assumed monthly pattern
    assert "Bigbasket" not in rec and "Uber" not in rec  # everyday spending needs 3+ to count as recurring


def test_duplicates_overlaps_price_increase_fees(bank_report):
    kinds = {}
    for i in bank_report.insights:
        kinds.setdefault(i.kind, []).append(i)
    dup = kinds["duplicate"][0]
    assert "Swiggy" in dup.title and dup.amount == 456 and dup.confidence == "high" and len(dup.evidence) == 2
    assert "music" in kinds["overlap"][0].title
    assert kinds["price_increase"][0].amount == 150  # Netflix 499 -> 649
    assert sum(i.amount for i in kinds["fee"]) == pytest.approx(489.7)
    assert kinds["unusual"][0].amount == 54990


def test_duplicate_window():
    rows = ["01/11/2026,UPI/ZOMATO/1,389,DR", "03/11/2026,UPI/ZOMATO/2,389,DR", "10/11/2026,UPI/ZOMATO/3,389,DR"]
    r = money.analyse(statements.parse_statement("x.csv", csv_bytes(rows)), AS_OF)
    dups = [i for i in r.insights if i.kind == "duplicate"]
    assert len(dups) == 1 and dups[0].confidence == "medium" and {e.date for e in dups[0].evidence} == {"2026-11-01", "2026-11-03"}


def test_card_bill_increase_compares_bill_to_bill(card_report):
    changes = {m.category: m for m in card_report.month_changes}
    assert changes["Shopping"].change == pytest.approx(5899) and changes["Shopping"].drivers[0].merchant == "Croma"
    assert changes["Fees & charges"].previous == 32 and changes["Fees & charges"].current == 2655


def test_upcoming_dues_and_card_due(bank_report, card_report):
    bank_due = [(o.name, o.due_date) for o in bank_report.upcoming]
    assert ("Rent", "2026-12-03") in bank_due and ("Bajaj Finserv", "2026-12-05") in bank_due
    assert all("2026-11-15" <= o.due_date <= "2026-12-30" for o in bank_report.upcoming)
    card_due = next(o for o in card_report.upcoming if o.kind == "Card due")
    assert card_due.amount == 38450 and card_due.due_date == "2026-12-05" and card_due.confidence == "high"
    assert card_report.card_summary["utilization_pct"] == 38.5


def test_cash_flow_projection(bank_report):
    cf = bank_report.cash_flow
    assert cf.balance == pytest.approx(44550.30) and cf.next_salary_date == "2026-12-07"
    assert cf.obligations_before_salary == pytest.approx(32340)  # electricity + rent + EMI before salary
    assert cf.status == "tight" and cf.projected_balance == pytest.approx(cf.balance - cf.obligations_before_salary - cf.everyday_spend_estimate)
    stmt = load("bank_statement_oct_nov_2026.csv")
    assert money.analyse(stmt, AS_OF, balance=20000).cash_flow.status == "shortfall"
    assert money.analyse(stmt, AS_OF, balance=200000).cash_flow.status == "ok"


def test_card_only_statement_asks_for_balance(card_report):
    assert card_report.cash_flow.status == "unknown" and "balance" in card_report.cash_flow.message.lower()


def test_action_plan_priorities_and_savings(bank_report):
    titles = [a.title for a in bank_report.actions]
    assert titles[0] == "Plan for the cash crunch"
    assert any("duplicate" in t for t in titles) and any("fees" in t for t in titles)
    assert bank_report.potential_monthly_savings == pytest.approx(456 + 489.7 + 99)


def test_every_insight_has_evidence_with_source(bank_report, card_report):
    for r in (bank_report, card_report):
        for i in r.insights:
            if i.kind != "utilization":
                assert i.evidence, i.title
            for e in i.evidence:
                assert e.date and e.merchant and e.amount > 0 and e.source and e.ref.startswith("S")
        assert r.disclaimer.startswith("These insights are informational only")


def test_explain_transaction():
    stmt = load("credit_card_nov_2026.pdf")
    fee = next(t for t in stmt.transactions if "LATE PAYMENT FEE" == t.description)
    facts = money.explain_transaction(stmt, fee.txn_id, AS_OF)
    assert facts["transaction"]["category"] == "Fees & charges" and "late" in facts["fee_explanation"].lower()
    with pytest.raises(Exception):
        money.explain_transaction(stmt, "T999", AS_OF)


def test_session_tools_give_the_model_evidence_ids():
    data = SessionData(statement=load("bank_statement_oct_nov_2026.csv"), as_of=AS_OF)
    tools = build_session_tools(data)
    assert {"money_report", "explain_transaction", "list_transactions"} <= set(tools) and "my_loan_preflight" not in tools
    out = tools["money_report"][1]()
    refs = {r for item in out["alerts"] for r in item["evidence"]}
    assert refs and refs <= set(out["evidence_index"]) and {c["ref"] for c in out["citations"]} == set(out["evidence_index"])
    listed = tools["list_transactions"][1](category="food")
    assert listed["count"] and all(c["kind"] == "statement" for c in listed["citations"])
    assert build_session_tools(SessionData()) == {}  # nothing uploaded -> no session tools
