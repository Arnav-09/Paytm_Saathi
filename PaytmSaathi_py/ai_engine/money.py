"""Money Clarity: deterministic analysis of a parsed statement (no LLM involved).

* category totals and month-over-month change (with the transactions that drove it)
* recurring charges (EMIs, premiums, subscriptions, bills, rent) and price increases
* duplicate charges, overlapping subscriptions, fees, unusually large debits
* upcoming obligations (from recurring patterns and the card's due date)
* cash-flow projection to the next salary (Safe / Tight / Shortfall)
* a prioritised action plan with potential savings

Every insight carries evidence (txn id, date, merchant, amount, source row/page) and a confidence
level. The LLM only explains these results; it never computes them.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import date, timedelta
from typing import Optional

from pydantic import BaseModel, Field

from .exceptions import CalculationInputError
from .statements import Statement, Transaction, rules
from .utils import inr

UPCOMING_WINDOW_DAYS = 45
_SPEND_EXCLUDE = {"Transfers", "Investments", "Money in", "Income", "Refunds & cashback"}
_OBLIGATION_KINDS = {"EMI & loans": "EMI", "Insurance": "Insurance premium", "Subscriptions": "Subscription",
                     "Rent & housing": "Rent", "Bills & utilities": "Bill"}
FEE_EXPLANATIONS = {
    "late": "A late-payment fee is charged when at least the minimum due is not paid by the due date. It is avoidable with autopay.",
    "finance charge": "Finance charges are interest on the unpaid card balance. Paying the full 'total amount due' by the due date avoids them.",
    "interest": "Interest is charged on the unpaid card balance when only part of the bill is paid.",
    "gst": "GST (18%) is charged on bank and card fees and interest, so every fee costs a little more than its face value.",
    "forex": "A foreign-currency markup (often 2-3.5%) is added to purchases in another currency or from foreign merchants.",
    "markup": "A foreign-currency markup (often 2-3.5%) is added to purchases in another currency or from foreign merchants.",
    "annual fee": "An annual card fee; many cards waive it if you spend a certain amount in a year.",
    "overlimit": "An over-limit fee is charged when spending goes above the card's credit limit.",
    "min bal": "A charge for not keeping the required average balance in a savings account.",
    "minimum balance": "A charge for not keeping the required average balance in a savings account.",
    "bounce": "A bounce/return charge applies when an EMI or cheque fails for lack of funds; it can also hurt your credit score.",
    "return": "A return charge applies when an auto-debit (NACH/ECS) fails for lack of funds; it can also hurt your credit score.",
    "sms": "A small charge for SMS alerts.",
    "fuel surcharge": "A fuel surcharge is added on fuel purchases; many cards refund it within limits.",
}


class Evidence(BaseModel):
    ref: str  # S1, S2 ... (citation id within a report)
    txn_id: str
    date: str
    merchant: str
    amount: float
    type: str
    source: str


class Insight(BaseModel):
    kind: str  # duplicate | fee | overlap | price_increase | unusual | utilization | card_due | ...
    title: str
    detail: str
    amount: Optional[float] = None
    confidence: str = "medium"  # high | medium | low
    evidence: list[Evidence] = Field(default_factory=list)


class CategoryTotal(BaseModel):
    category: str
    total: float
    count: int
    share_pct: float


class MonthChange(BaseModel):
    category: str
    previous: float
    current: float
    change: float
    change_pct: Optional[float]
    drivers: list[Evidence] = Field(default_factory=list)


class RecurringCharge(BaseModel):
    merchant: str
    category: str
    average_amount: float
    frequency: str  # weekly | monthly | yearly | monthly (assumed) | yearly (assumed)
    occurrences: int
    last_date: str
    next_expected: Optional[str]
    confidence: str
    evidence: list[Evidence] = Field(default_factory=list)


class Obligation(BaseModel):
    name: str
    kind: str
    due_date: str
    amount: float
    confidence: str
    basis: str
    evidence: list[Evidence] = Field(default_factory=list)


class CashFlow(BaseModel):
    as_of: str
    status: str  # ok | tight | shortfall | unknown
    balance: Optional[float] = None
    balance_basis: str = ""
    next_salary_date: Optional[str] = None
    salary_amount: Optional[float] = None
    obligations_before_salary: float = 0.0
    everyday_spend_estimate: float = 0.0
    projected_balance: Optional[float] = None
    message: str = ""
    assumptions: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)


class Action(BaseModel):
    priority: int
    title: str
    detail: str
    saving: Optional[float] = None
    evidence: list[Evidence] = Field(default_factory=list)


class MoneyReport(BaseModel):
    source: str
    kind: str
    period_start: Optional[str]
    period_end: Optional[str]
    as_of: str
    total_in: float
    total_out: float
    categories: list[CategoryTotal] = Field(default_factory=list)
    month_changes: list[MonthChange] = Field(default_factory=list)
    recurring: list[RecurringCharge] = Field(default_factory=list)
    insights: list[Insight] = Field(default_factory=list)
    upcoming: list[Obligation] = Field(default_factory=list)
    cash_flow: CashFlow
    actions: list[Action] = Field(default_factory=list)
    potential_monthly_savings: float = 0.0
    card_summary: dict = Field(default_factory=dict)
    disclaimer: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _d(s: str) -> date:
    return date.fromisoformat(s)


class _Refs:
    """Stable citation ids (S1, S2 ...) per transaction within one report."""

    def __init__(self):
        self.ids: dict[str, str] = {}

    def ev(self, t: Transaction) -> Evidence:
        ref = self.ids.setdefault(t.txn_id, f"S{len(self.ids) + 1}")
        return Evidence(ref=ref, txn_id=t.txn_id, date=t.date, merchant=t.merchant, amount=t.amount, type=t.type, source=t.source)


def _close(a: float, b: float) -> bool:
    cfg = rules()["recurring"]
    return abs(a - b) <= max(cfg["amount_tolerance_abs"], cfg["amount_tolerance_pct"] / 100 * max(a, b))


def _frequency(interval: float) -> Optional[tuple[str, int]]:
    cfg = rules()["recurring"]
    for name, (lo, hi), step in (("weekly", cfg["weekly_days"], 7), ("monthly", cfg["monthly_days"], 30),
                                 ("yearly", cfg["yearly_days"], 365)):
        if lo <= interval <= hi:
            return name, step
    return None


def _add_month(d: date, anchor_day: int) -> date:
    year, month = (d.year + 1, 1) if d.month == 12 else (d.year, d.month + 1)
    days_in_month = [31, 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28,
                     31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
    return date(year, month, min(anchor_day, days_in_month))


def _advance(last: date, frequency: str, interval_days: int, as_of: date) -> date:
    """Next expected date after ``last`` that is on or after ``as_of`` (keeps the same day of month)."""
    def step(d: date) -> date:
        return _add_month(d, last.day) if frequency.startswith("monthly") else d + timedelta(days=interval_days)

    nxt = step(last)
    while nxt < as_of:
        nxt = step(nxt)
    return nxt


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------
def category_totals(txns: list[Transaction]) -> list[CategoryTotal]:
    spend = [t for t in txns if t.type == "debit" and t.category not in _SPEND_EXCLUDE]
    total = sum(t.amount for t in spend) or 1.0
    groups: dict[str, list[Transaction]] = defaultdict(list)
    for t in spend:
        groups[t.category].append(t)
    out = [CategoryTotal(category=c, total=round(sum(t.amount for t in g), 2), count=len(g),
                         share_pct=round(100 * sum(t.amount for t in g) / total, 1)) for c, g in groups.items()]
    return sorted(out, key=lambda c: -c.total)


def month_changes(txns: list[Transaction], refs: _Refs) -> list[MonthChange]:
    spend = [t for t in txns if t.type == "debit" and t.category not in _SPEND_EXCLUDE]
    period = lambda t: t.period or t.date[:7]  # noqa: E731 - card bills compare bill-to-bill
    months = sorted({period(t) for t in spend})
    if len(months) < 2:
        return []
    prev_m, cur_m = months[-2], months[-1]
    out = []
    for cat in sorted({t.category for t in spend}):
        prev = [t for t in spend if t.category == cat and period(t) == prev_m]
        cur = [t for t in spend if t.category == cat and period(t) == cur_m]
        p, c = round(sum(t.amount for t in prev), 2), round(sum(t.amount for t in cur), 2)
        if p == c:
            continue
        prev_merchants = {t.merchant_key for t in prev}
        drivers = sorted([t for t in cur if t.merchant_key not in prev_merchants or t.duplicate or t.fee],
                         key=lambda t: -t.amount)[:3] or sorted(cur, key=lambda t: -t.amount)[:2]
        out.append(MonthChange(category=cat, previous=p, current=c, change=round(c - p, 2),
                               change_pct=round(100 * (c - p) / p, 1) if p else None,
                               drivers=[refs.ev(t) for t in drivers]))
    return sorted(out, key=lambda m: -m.change)


def detect_recurring(txns: list[Transaction], as_of: date, refs: _Refs) -> list[RecurringCharge]:
    """Same merchant, similar amount, regular interval. Single known-subscription/EMI/premium items are
    reported as 'assumed' with low confidence."""
    out = []
    debits: dict[str, list[Transaction]] = defaultdict(list)
    for t in txns:
        if t.type == "debit" and t.category not in ("Transfers", "Cash withdrawal", "Fees & charges"):
            debits[t.merchant_key].append(t)
    for key, items in debits.items():
        items.sort(key=lambda t: t.date)
        median_amt = statistics.median(t.amount for t in items)
        same = [t for t in items if _close(t.amount, median_amt)]
        freq = None
        # Bills/EMIs/subscriptions repeat by nature: 2 matching payments are enough. Everyday spending
        # (groceries, cabs) needs 3+ before we call it recurring.
        needed = 2 if items[-1].category in _OBLIGATION_KINDS else 3
        if len(same) >= needed:
            gaps = [(_d(b.date) - _d(a.date)).days for a, b in zip(same, same[1:])]
            gaps = [g for g in gaps if g > 3]  # ignore same-week duplicates
            if gaps:
                freq = _frequency(statistics.median(gaps))
        if freq:
            name, step = freq
            confidence = "high" if len(same) >= 3 else "medium"
        elif items[-1].category in _OBLIGATION_KINDS:
            name = "yearly (assumed)" if items[-1].category == "Insurance" and len(items) == 1 else "monthly (assumed)"
            step = 365 if name.startswith("yearly") else 30
            same, confidence = [items[-1]], "low"
        else:
            continue
        for t in same:
            t.recurring = True
        last = _d(same[-1].date)
        nxt = _advance(last, name, step, as_of)
        out.append(RecurringCharge(merchant=same[-1].merchant, category=same[-1].category,
                                   average_amount=round(statistics.mean(t.amount for t in same), 2), frequency=name,
                                   occurrences=len(same), last_date=same[-1].date, next_expected=nxt.isoformat(),
                                   confidence=confidence, evidence=[refs.ev(t) for t in same[-3:]]))
    return sorted(out, key=lambda r: -r.average_amount)


def detect_duplicates(txns: list[Transaction], refs: _Refs) -> list[Insight]:
    window = rules()["duplicate_window_days"]
    debits = sorted([t for t in txns if t.type == "debit" and t.category not in ("Transfers",)], key=lambda t: t.date)
    out, used = [], set()
    for i, a in enumerate(debits):
        for b in debits[i + 1:]:
            gap = (_d(b.date) - _d(a.date)).days
            if gap > window:
                break
            if b.txn_id in used or a.txn_id in used:
                continue
            if a.merchant_key == b.merchant_key and abs(a.amount - b.amount) < 1:
                a.duplicate = b.duplicate = True
                used |= {a.txn_id, b.txn_id}
                conf = "high" if gap == 0 else "medium"
                out.append(Insight(kind="duplicate", title=f"Possible duplicate charge at {a.merchant}",
                                   detail=f"{inr(a.amount, 2)} was charged twice {'on the same day' if gap == 0 else f'{gap} day(s) apart'} "
                                          f"({a.date} and {b.date}). If you made only one purchase, ask the merchant or bank for a refund.",
                                   amount=a.amount, confidence=conf, evidence=[refs.ev(a), refs.ev(b)]))
    return out


def detect_overlaps(recurring: list[RecurringCharge]) -> list[Insight]:
    out = []
    for group, names in rules()["subscription_groups"].items():
        hits = [r for r in recurring if r.category == "Subscriptions" and any(n in r.merchant.lower() for n in names)]
        if len(hits) >= 2:
            cheapest = min(hits, key=lambda r: r.average_amount)
            out.append(Insight(kind="overlap", title=f"{len(hits)} {group.lower()} subscriptions",
                               detail=f"You pay for {', '.join(h.merchant for h in hits)}. Keeping one could save about "
                                      f"{inr(cheapest.average_amount)} a month.",
                               amount=cheapest.average_amount, confidence="medium",
                               evidence=[e for h in hits for e in h.evidence[-1:]]))
    return out


def detect_price_increases(txns: list[Transaction], refs: _Refs) -> list[Insight]:
    out = []
    groups: dict[str, list[Transaction]] = defaultdict(list)
    for t in txns:
        if t.type == "debit" and t.category in ("Subscriptions", "Insurance", "Bills & utilities") and not t.duplicate:
            groups[t.merchant_key].append(t)
    for items in groups.values():
        items.sort(key=lambda t: t.date)
        if len(items) >= 2 and items[-2].date[:7] != items[-1].date[:7]:
            old, new = items[-2], items[-1]
            if new.amount > old.amount * 1.05 and new.amount - old.amount >= 20:
                out.append(Insight(kind="price_increase", title=f"{new.merchant} costs more now",
                                   detail=f"Went up from {inr(old.amount, 2)} ({old.date}) to {inr(new.amount, 2)} ({new.date}).",
                                   amount=round(new.amount - old.amount, 2), confidence="high",
                                   evidence=[refs.ev(old), refs.ev(new)]))
    return out


def detect_fees(txns: list[Transaction], refs: _Refs) -> list[Insight]:
    fees = [t for t in txns if t.fee and t.type == "debit"]
    out = []
    for t in fees:
        low = t.description.lower()
        why = next((v for k, v in FEE_EXPLANATIONS.items() if k in low), "A bank or card charge.")
        out.append(Insight(kind="fee", title=f"Fee: {t.description[:48]}", detail=why, amount=t.amount,
                           confidence="high", evidence=[refs.ev(t)]))
    return out


def detect_unusual(txns: list[Transaction], refs: _Refs) -> list[Insight]:
    debits = [t for t in txns if t.type == "debit" and t.category not in ("Transfers", "Investments", "EMI & loans", "Rent & housing")]
    if len(debits) < 5:
        return []
    median = statistics.median(t.amount for t in debits)
    threshold = max(3 * median, 10000)
    return [Insight(kind="unusual", title=f"Large payment to {t.merchant}",
                    detail=f"{inr(t.amount, 2)} on {t.date} is much larger than your usual spend (typical {inr(median)}). "
                           "Check that you recognise it.", amount=t.amount, confidence="low", evidence=[refs.ev(t)])
            for t in debits if t.amount >= threshold and not t.recurring][:5]


def upcoming_obligations(stmt: Statement, recurring: list[RecurringCharge], as_of: date) -> list[Obligation]:
    end = as_of + timedelta(days=UPCOMING_WINDOW_DAYS)
    out = []
    for r in recurring:
        if r.category not in _OBLIGATION_KINDS or not r.next_expected:
            continue
        due = _d(r.next_expected)
        if as_of <= due <= end:
            out.append(Obligation(name=r.merchant, kind=_OBLIGATION_KINDS[r.category], due_date=due.isoformat(),
                                  amount=r.average_amount, confidence=r.confidence,
                                  basis=f"repeats {r.frequency}; last paid {r.last_date}", evidence=r.evidence[-1:]))
    if stmt.card_due_date and stmt.card_total_due:
        due = _d(stmt.card_due_date)
        if due >= as_of:
            out.append(Obligation(name="Credit card bill", kind="Card due", due_date=due.isoformat(),
                                  amount=stmt.card_total_due, confidence="high",
                                  basis=f"from your statement (minimum due {inr((stmt.card_minimum_due or 0), 2)})",
                                  evidence=[Evidence(ref="S0", txn_id="-", date=stmt.card_due_date, merchant="Card statement",
                                                     amount=stmt.card_total_due, type="due", source=f"{stmt.source_filename} (summary)")]))
    return sorted(out, key=lambda o: o.due_date)


def project_cash_flow(stmt: Statement, txns: list[Transaction], upcoming: list[Obligation], as_of: date,
                      refs: _Refs, balance: Optional[float] = None, next_salary: Optional[date] = None,
                      one_off_ids: Optional[set] = None) -> CashFlow:
    cfg = rules()["cash_flow"]
    cf = CashFlow(as_of=as_of.isoformat(), status="unknown")
    if balance is not None:
        cf.balance, cf.balance_basis = balance, "entered by you"
    elif stmt.kind == "bank" and stmt.closing_balance is not None:
        cf.balance, cf.balance_basis = stmt.closing_balance, f"closing balance on your statement ({stmt.period_end})"
    salaries = sorted([t for t in txns if t.type == "credit" and t.category == "Income"], key=lambda t: t.date)
    if salaries:
        cf.salary_amount = round(statistics.median(t.amount for t in salaries), 2)
        cf.evidence.append(refs.ev(salaries[-1]))
    if next_salary is None and salaries:
        next_salary = _advance(_d(salaries[-1].date), "monthly", 30, as_of)
    if next_salary is None:
        next_salary = as_of + timedelta(days=30)
        cf.assumptions.append("No salary credit found, so the next 30 days are checked.")
    cf.next_salary_date = next_salary.isoformat()
    if cf.balance is None:
        cf.message = "Add your current bank balance to check for a cash crunch." if stmt.kind == "card" else \
            "The statement has no balance column; add your current balance to check for a cash crunch."
        return cf
    due = [o for o in upcoming if as_of <= _d(o.due_date) < next_salary]
    cf.obligations_before_salary = round(sum(o.amount for o in due), 2)
    for o in due:
        cf.evidence += o.evidence
    span = max((_d(stmt.period_end) - _d(stmt.period_start)).days + 1, 1) if stmt.period_start else 30
    one_off_ids = one_off_ids or set()
    everyday = [t for t in txns if t.type == "debit" and not t.recurring and not t.duplicate and t.txn_id not in one_off_ids
                and t.category not in _SPEND_EXCLUDE | {"Fees & charges", "Cash withdrawal"}]
    daily = sum(t.amount for t in everyday) / span
    days = max((next_salary - as_of).days, 0)
    cf.everyday_spend_estimate = round(daily * days, 2)
    cf.projected_balance = round(cf.balance - cf.obligations_before_salary - cf.everyday_spend_estimate, 2)
    cf.assumptions.append(f"Everyday spending of about {inr(daily)}/day continues (your average in this statement, "
                          "leaving out one-off large purchases and duplicates).")
    tight_at = max(cfg["tight_buffer_min_rupees"], cfg["tight_buffer_pct_of_income"] / 100 * (cf.salary_amount or 0))
    if cf.projected_balance < 0:
        cf.status = "shortfall"
        cf.message = (f"You may run short by about {inr(-cf.projected_balance)} before your next salary on {cf.next_salary_date}.")
    elif cf.projected_balance < tight_at:
        cf.status = "tight"
        cf.message = f"Money looks tight: about {inr(cf.projected_balance)} may be left before {cf.next_salary_date}."
    else:
        cf.status = "ok"
        cf.message = f"You should have about {inr(cf.projected_balance)} left before {cf.next_salary_date}."
    return cf


def _card_summary(stmt: Statement, refs: _Refs) -> tuple[dict, list[Insight]]:
    if stmt.kind != "card":
        return {}, []
    summary = {k: getattr(stmt, k) for k in ("card_total_due", "card_minimum_due", "card_due_date", "card_credit_limit")}
    insights = []
    if stmt.card_total_due and stmt.card_credit_limit:
        util = round(100 * stmt.card_total_due / stmt.card_credit_limit, 1)
        summary["utilization_pct"] = util
        if util > 30:
            insights.append(Insight(kind="utilization", title=f"Card usage is {util}% of your limit",
                                    detail="Using more than about 30% of your limit can lower your credit score.",
                                    amount=stmt.card_total_due, confidence="high"))
    return summary, insights


def build_actions(cf: CashFlow, upcoming: list[Obligation], insights: list[Insight], changes: list[MonthChange],
                  stmt: Statement) -> tuple[list[Action], float]:
    actions: list[Action] = []
    if cf.status in ("shortfall", "tight"):
        actions.append(Action(priority=0, title="Plan for the cash crunch", detail=cf.message + " Pay essential dues first "
                              "and delay non-essential spending until salary arrives.", evidence=cf.evidence[:3]))
    card = next((o for o in upcoming if o.kind == "Card due"), None)
    if card:
        actions.append(Action(priority=0, title=f"Pay the card bill of {inr(card.amount)} by {card.due_date}",
                              detail="Pay the total amount due, not just the minimum, to avoid interest and late fees.",
                              evidence=card.evidence))
    savings = 0.0
    for i in insights:
        if i.kind == "duplicate":
            actions.append(Action(priority=1, title=f"Dispute the duplicate {inr(i.amount)} at {i.title.split(' at ')[-1]}",
                                  detail="Ask the merchant for a refund; if they don't respond, raise it in Payments & KYC → Payment problem.",
                                  saving=i.amount, evidence=i.evidence))
            savings += i.amount or 0
    fee_total = sum(i.amount or 0 for i in insights if i.kind == "fee")
    if fee_total:
        actions.append(Action(priority=1, title=f"Avoid {inr(fee_total)} in fees and charges",
                              detail="Set up autopay for the full card bill and EMIs, and keep the minimum balance.",
                              saving=round(fee_total, 2), evidence=[e for i in insights if i.kind == "fee" for e in i.evidence][:4]))
        savings += fee_total
    for i in insights:
        if i.kind == "overlap":
            actions.append(Action(priority=2, title=f"Review {i.title}", detail=i.detail, saving=i.amount, evidence=i.evidence))
            savings += i.amount or 0
        elif i.kind == "price_increase":
            actions.append(Action(priority=3, title=i.title, detail=i.detail + " Check whether a cheaper plan suits you.", evidence=i.evidence))
        elif i.kind == "utilization":
            actions.append(Action(priority=2, title="Bring card usage below 30% of the limit", detail=i.detail, evidence=i.evidence))
    for m in changes[:1]:
        if m.change > 0:
            actions.append(Action(priority=3, title=f"{m.category} spending went up by {inr(m.change)}",
                                  detail="Set a monthly budget for this category.", evidence=m.drivers))
    actions.sort(key=lambda a: a.priority)
    return actions, round(savings, 2)


def analyse(stmt: Statement, as_of: Optional[date] = None, balance: Optional[float] = None,
            next_salary: Optional[date] = None) -> MoneyReport:
    """Full Money Clarity report for a statement."""
    if not stmt.transactions:
        raise CalculationInputError("The statement has no transactions.")
    if balance is not None and balance < 0:
        raise CalculationInputError("Balance cannot be negative.")
    as_of = as_of or date.today()
    refs = _Refs()
    txns = [t.model_copy() for t in stmt.transactions]
    recurring = detect_recurring(txns, as_of, refs)
    duplicates = detect_duplicates(txns, refs)
    insights = duplicates + detect_overlaps(recurring) + detect_price_increases(txns, refs) + detect_fees(txns, refs) \
        + detect_unusual(txns, refs)
    card_summary, card_insights = _card_summary(stmt, refs)
    insights += card_insights
    changes = month_changes(txns, refs)
    upcoming = upcoming_obligations(stmt, recurring, as_of)
    one_offs = {e.txn_id for i in insights if i.kind == "unusual" for e in i.evidence}
    cf = project_cash_flow(stmt, txns, upcoming, as_of, refs, balance, next_salary, one_offs)
    actions, savings = build_actions(cf, upcoming, insights, changes, stmt)
    return MoneyReport(
        source=stmt.source_filename, kind=stmt.kind, period_start=stmt.period_start, period_end=stmt.period_end,
        as_of=as_of.isoformat(), total_in=round(sum(t.amount for t in txns if t.type == "credit"), 2),
        total_out=round(sum(t.amount for t in txns if t.type == "debit"), 2), categories=category_totals(txns),
        month_changes=changes, recurring=recurring, insights=insights, upcoming=upcoming, cash_flow=cf,
        actions=actions, potential_monthly_savings=savings, card_summary=card_summary,
        disclaimer=rules()["disclaimer"])


def explain_transaction(stmt: Statement, txn_id: str, as_of: Optional[date] = None) -> dict:
    """Facts about one transaction for the assistant to explain (pattern, duplicates, fee meaning, comparison)."""
    report = analyse(stmt, as_of)
    txns = {t.txn_id: t for t in stmt.transactions}
    if txn_id not in txns:
        raise CalculationInputError(f"No transaction {txn_id} in this statement.")
    t = txns[txn_id]
    same_cat = [x.amount for x in stmt.transactions if x.category == t.category and x.type == t.type]
    facts = {
        "transaction": t.model_dump(include={"txn_id", "date", "description", "merchant", "amount", "type", "category", "source"}),
        "category_average": round(statistics.mean(same_cat), 2) if same_cat else None,
        "times_seen_this_merchant": sum(1 for x in stmt.transactions if x.merchant_key == t.merchant_key),
        "recurring": next((r.model_dump(exclude={"evidence"}) for r in report.recurring if any(e.txn_id == txn_id for e in r.evidence)), None),
        "flags": [i.model_dump(exclude={"evidence"}) for i in report.insights if any(e.txn_id == txn_id for e in i.evidence)],
    }
    low = t.description.lower()
    facts["fee_explanation"] = next((v for k, v in FEE_EXPLANATIONS.items() if k in low), None) if t.fee else None
    return facts
