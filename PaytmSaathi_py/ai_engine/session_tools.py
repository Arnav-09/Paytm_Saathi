"""Chat tools that use the user's SESSION data (uploaded statements, saved loan details, documents).

These run inside the app process, not over MCP: the data lives only in the user's session (it is never
written to disk), and the separate MCP server process cannot - and should not - see it.

Every result carries citation ids:
  [S#] -> a statement row/page (date, merchant, amount, source)
  [D#] -> a line in an uploaded document (rejection letter, credit report)
The assistant shows only the citations the answer actually uses.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional

from . import lending, money
from .exceptions import PaytmSaathiError
from .statements import Statement
from .utils import inr


@dataclass
class SessionData:
    statement: Optional[Statement] = None
    loan_inputs: Optional[lending.LoanPreflightInput] = None
    documents: list[tuple[str, list[tuple[str, str]]]] = field(default_factory=list)  # (name, [(location, line)])
    as_of: date = field(default_factory=date.today)
    balance: Optional[float] = None
    next_salary: Optional[date] = None


def _cite_statement(ev: money.Evidence) -> dict:
    return {"ref": ev.ref, "kind": "statement", "title": ev.merchant,
            "detail": f"{ev.date} · {inr(ev.amount, 2)} · {ev.source}", "text": ev.source}


def _refs(evidence: list[money.Evidence]) -> list[str]:
    return [e.ref for e in evidence]


def compact_report(r: money.MoneyReport) -> dict:
    """What the model sees: numbers + evidence ids only (no raw descriptions beyond merchants)."""
    every: dict[str, money.Evidence] = {}

    def keep(evs):
        for e in evs:
            every[e.ref] = e
        return _refs(evs)

    out = {
        "statement": {"source": r.source, "kind": r.kind, "period": f"{r.period_start} to {r.period_end}", "as_of": r.as_of},
        "totals": {"money_in": r.total_in, "money_out": r.total_out},
        "top_categories": [c.model_dump() for c in r.categories[:6]],
        "bill_change": [{"category": m.category, "previous": m.previous, "current": m.current, "change": m.change,
                         "change_pct": m.change_pct, "drivers": keep(m.drivers)} for m in r.month_changes[:5]],
        "recurring": [{"merchant": x.merchant, "category": x.category, "amount": x.average_amount, "frequency": x.frequency,
                       "next_expected": x.next_expected, "confidence": x.confidence, "evidence": keep(x.evidence[-2:])}
                      for x in r.recurring[:10]],
        "alerts": [{"kind": i.kind, "title": i.title, "detail": i.detail, "amount": i.amount, "confidence": i.confidence,
                    "evidence": keep(i.evidence[:3])} for i in r.insights[:12]],
        "upcoming_dues": [{"name": o.name, "kind": o.kind, "due_date": o.due_date, "amount": o.amount,
                           "confidence": o.confidence, "basis": o.basis, "evidence": keep(o.evidence)} for o in r.upcoming[:10]],
        "cash_flow": {**r.cash_flow.model_dump(exclude={"evidence"}), "evidence": keep(r.cash_flow.evidence[:5])},
        "action_plan": [{"title": a.title, "detail": a.detail, "saving": a.saving, "evidence": keep(a.evidence[:3])}
                        for a in r.actions[:7]],
        "potential_monthly_savings": r.potential_monthly_savings,
        "card": r.card_summary,
        "disclaimer": r.disclaimer,
    }
    out["evidence_index"] = {ref: f"{e.date} · {e.merchant} · {inr(e.amount, 2)} · {e.source}" for ref, e in every.items()}
    out["citations"] = [_cite_statement(e) for e in every.values()]
    return out


def compact_preflight(r: lending.LoanPreflightResult) -> dict:
    return {
        "inputs": r.inputs.model_dump(), "emi": r.emi, "total_interest": r.total_interest, "total_payment": r.total_payment,
        "dti_before_pct": r.dti_before_pct, "dti_after_pct": r.dti_after_pct,
        "monthly_leftover_before": r.before.buffer, "monthly_leftover_after": r.after.buffer,
        "status": lending.STATUS_LABELS[r.status], "status_reason": r.status_reason,
        "affordable_loan_amount": r.affordable_loan_amount,
        "stress_test": [s.model_dump() for s in r.scenarios],
        "risk_factors": [f.model_dump() for f in r.risk_factors], "actions": r.actions,
        "documents_needed": [d["en"] for d in r.documents], "explanations": r.explanations,
        "assumptions": r.assumptions, "grounding": r.grounding, "disclaimer": r.disclaimer,
    }


def build_session_tools(data: SessionData) -> dict[str, tuple[dict, Callable[..., dict]]]:
    """name -> (OpenAI tool schema, function). Only tools whose data exists are offered."""
    tools: dict[str, tuple[dict, Callable[..., dict]]] = {}

    def schema(name: str, description: str, props: dict, required: Optional[list] = None) -> dict:
        return {"type": "function", "function": {"name": name, "description": description,
                                                 "parameters": {"type": "object", "properties": props, "required": required or []}}}

    if data.statement is not None:
        stmt = data.statement

        def money_report(current_balance: Optional[float] = None, next_salary_date: Optional[str] = None) -> dict:
            salary = date.fromisoformat(next_salary_date) if next_salary_date else data.next_salary
            bal = current_balance if current_balance is not None else data.balance
            return compact_report(money.analyse(stmt, data.as_of, bal, salary))

        def explain_transaction(txn_id: str) -> dict:
            report = money.analyse(stmt, data.as_of, data.balance, data.next_salary)
            facts = money.explain_transaction(stmt, txn_id.strip().upper(), data.as_of)
            t = next(x for x in stmt.transactions if x.txn_id == txn_id.strip().upper())
            ev = money.Evidence(ref="S1", txn_id=t.txn_id, date=t.date, merchant=t.merchant, amount=t.amount, type=t.type, source=t.source)
            facts["evidence_ref"] = "S1"
            facts["citations"] = [_cite_statement(ev)]
            facts["disclaimer"] = report.disclaimer
            return facts

        def list_transactions(category: Optional[str] = None, merchant: Optional[str] = None, limit: int = 15) -> dict:
            rows = [t for t in stmt.transactions
                    if (not category or category.lower() in t.category.lower())
                    and (not merchant or merchant.lower() in t.merchant.lower() or merchant.lower() in t.description.lower())]
            rows = rows[-max(1, min(int(limit), 40)):]
            cites, items = [], []
            for i, t in enumerate(rows, 1):
                ev = money.Evidence(ref=f"S{i}", txn_id=t.txn_id, date=t.date, merchant=t.merchant, amount=t.amount, type=t.type, source=t.source)
                cites.append(_cite_statement(ev))
                items.append({"ref": f"S{i}", "txn_id": t.txn_id, "date": t.date, "merchant": t.merchant, "amount": t.amount,
                              "type": t.type, "category": t.category})
            return {"transactions": items, "total": round(sum(t.amount for t in rows if t.type == "debit"), 2),
                    "count": len(rows), "citations": cites}

        tools["money_report"] = (schema(
            "money_report",
            "Analyse the user's uploaded statement: spending by category, why the bill changed, recurring charges and "
            "subscriptions, duplicate charges, fees, upcoming bills/EMIs/premiums/card due dates, cash-flow risk before next "
            "salary, and a prioritised action plan. Every fact has evidence ids like S3.",
            {"current_balance": {"type": "number", "description": "user's current bank balance, only if they told you"},
             "next_salary_date": {"type": "string", "description": "YYYY-MM-DD, only if the user told you"}}), money_report)
        tools["explain_transaction"] = (schema(
            "explain_transaction", "Facts about one transaction (by id such as T12): pattern, duplicates, fee meaning, comparison.",
            {"txn_id": {"type": "string"}}, ["txn_id"]), explain_transaction)
        tools["list_transactions"] = (schema(
            "list_transactions", "List the user's transactions, optionally filtered by category or merchant.",
            {"category": {"type": "string"}, "merchant": {"type": "string"}, "limit": {"type": "integer"}}), list_transactions)

    if data.loan_inputs is not None:
        inp = data.loan_inputs

        def my_loan_preflight(shock_amount: Optional[float] = None) -> dict:
            return compact_preflight(lending.loan_preflight(inp, shock_amount))

        def my_loan_what_if(tenure_months: Optional[int] = None, loan_amount: Optional[float] = None,
                            annual_rate_pct: Optional[float] = None, shock_amount: Optional[float] = None,
                            income_drop_pct: Optional[float] = None) -> dict:
            return lending.loan_what_if(inp, tenure_months, loan_amount, annual_rate_pct, shock_amount, income_drop_pct)

        tools["my_loan_preflight"] = (schema(
            "my_loan_preflight",
            "Loan Pre-Flight for the loan details the user saved: EMI, total interest, debt-to-income, monthly leftover, "
            "Safe/Tight/High Risk stress test, risk factors, actions, documents and explanations.",
            {"shock_amount": {"type": "number", "description": "one-time emergency expense in rupees, e.g. hospitalisation"}}),
            my_loan_preflight)
        tools["my_loan_what_if"] = (schema(
            "my_loan_what_if", "Compare the saved loan plan with a change: tenure, amount, rate, emergency expense or income drop.",
            {"tenure_months": {"type": "integer"}, "loan_amount": {"type": "number"}, "annual_rate_pct": {"type": "number"},
             "shock_amount": {"type": "number"}, "income_drop_pct": {"type": "number"}}), my_loan_what_if)

    if data.documents:
        def decode_my_documents() -> dict:
            reasons, cites, i = [], [], 0
            for name, lines in data.documents:
                decoded = lending.decode_rejection(lines, name)
                for r in decoded.reasons:
                    i += 1
                    ref = f"D{i}"
                    reasons.append({**r.model_dump(), "ref": ref})
                    cites.append({"ref": ref, "kind": "document", "title": name, "detail": r.location, "text": r.quote})
            return {"reasons": reasons, "message": "" if reasons else "No common rejection reason recognised.",
                    "citations": cites, "disclaimer": lending.preflight_rules()["disclaimer"]}

        tools["decode_my_documents"] = (schema(
            "decode_my_documents", "Decode the user's uploaded loan rejection letter or credit report into plain reasons "
            "and next steps, quoting the exact lines (ids D1, D2...).", {}), decode_my_documents)
    return tools


# When the user's question clearly needs their own data, the app runs these tools itself before asking the model,
# so the answer is grounded even if the model would not have called the tool.
AUTO_INTENTS = {
    "money_report": re.compile(r"\b(?:bill|bills|spend|spent|spending|subscriptions?|duplicates?|dues?|cash|crunch|salary|fees?|"
                               r"charges?|transactions?|statements?|budget|savings?|recurring|upcoming|increased?|money|expenses?)\b", re.I),
    "my_loan_preflight": re.compile(r"\b(?:loan|emi|afford|tenure|borrow|hospitali[sz]\w*|emergency)\b", re.I),
    "decode_my_documents": re.compile(r"\b(?:reject\w*|declined|denied|refused|credit report|why .{0,40}(?:loan|application))\b", re.I),
}


def safe_call(fn: Callable[..., dict], args: dict) -> dict:
    try:
        params = inspect.signature(fn).parameters  # ignore arguments the tool does not take
        return fn(**{k: v for k, v in (args or {}).items() if k in params})
    except PaytmSaathiError as exc:
        return {"error": exc.message}
    except (TypeError, ValueError, StopIteration) as exc:
        return {"error": f"Invalid input: {exc}" if str(exc) else "Not found."}
