"""Creates differently-formatted statements for testing the layout-tolerant parser (samples/statements/variants/).

    python samples/make_statement_variants.py

Every bank file holds the SAME 12 November transactions (BANK below) laid out the way different banks export them,
so each one must read back to exactly the same transactions. The card files hold the same 10 card transactions.
expected.json records what each file must produce (count, totals, balances, card summary, spot checks); it is
computed here from the source data, not from the parser, and tests/test_statement_variants.py checks it.

All names, numbers and banks are fictional ("SAMPLE BANK"); the layouts are only modelled on common export styles.
"""

import csv
import io
import json
import pathlib
from datetime import date, datetime

import fitz  # PyMuPDF
from openpyxl import Workbook

OUT = pathlib.Path(__file__).resolve().parent / "statements" / "variants"

D, C = "debit", "credit"
BANK = [  # (date, description, amount, type)
    (date(2026, 11, 1), "UPI/RENT NOV/RAMESH OWNER/403011554321", 18000.00, D),
    (date(2026, 11, 3), "NACH DR BAJAJ FINSERV LTD EMI 4XX88", 12500.00, D),
    (date(2026, 11, 5), "NEFT CR SALARY ACME TECHNOLOGIES PVT LTD", 85000.00, C),
    (date(2026, 11, 6), "POS NETFLIX.COM MUMBAI", 649.00, D),
    (date(2026, 11, 8), "UPI/SWIGGY/403011223344/Food order", 456.00, D),
    (date(2026, 11, 9), "UPI/BIGBASKET/403055667788/Groceries", 2675.50, D),
    (date(2026, 11, 10), "IMPS/P2A/FROM PRIYA SHARMA/SPLIT DINNER", 1200.00, C),
    (date(2026, 11, 11), "BESCOM ELECTRICITY BILL PAYMENT", 1840.00, D),
    (date(2026, 11, 12), "ATM CASH WITHDRAWAL MG ROAD", 5000.00, D),
    (date(2026, 11, 13), "REFUND FLIPKART ORDER OD4021", 999.00, C),
    (date(2026, 11, 14), "MIN BAL CHARGES OCT", 400.00, D),
    (date(2026, 11, 14), "GST ON MIN BAL CHARGES", 72.00, D),
]
OPENING = 32345.60

CARD = [
    (date(2026, 10, 18), "AMAZON SELLER SERVICES", 3999.00, D),
    (date(2026, 10, 21), "ZOMATO", 389.00, D),
    (date(2026, 10, 25), "INDIAN OIL FUEL STATION", 3400.00, D),
    (date(2026, 10, 29), "CROMA RETAIL ELECTRONICS", 12999.00, D),
    (date(2026, 11, 2), "SWIGGY", 720.00, D),
    (date(2026, 11, 4), "REFUND AMAZON SELLER SERVICES", 1200.00, C),
    (date(2026, 11, 6), "LATE PAYMENT FEE", 1000.00, D),
    (date(2026, 11, 6), "GST ON LATE PAYMENT FEE", 180.00, D),
    (date(2026, 11, 8), "BBPS PAYMENT 7788", 5000.00, C),
    (date(2026, 11, 12), "FINANCE CHARGES", 1250.00, D),
]
CARD_SUMMARY = {"card_total_due": 38450.0, "card_minimum_due": 1930.0, "card_due_date": "2026-12-05",
                "card_credit_limit": 100000.0, "period_start": "2026-10-16", "period_end": "2026-11-15"}


def balances(txns, opening):
    out, bal = [], opening
    for _, _, amt, typ in txns:
        bal = round(bal - amt if typ == D else bal + amt, 2)
        out.append(bal)
    return out


BAL = balances(BANK, OPENING)
CLOSING = BAL[-1]


def indian(n: float, decimals: int = 2) -> str:
    """1234567.5 -> '12,34,567.50' (Indian digit grouping)."""
    whole, _, frac = f"{abs(n):.{decimals}f}".partition(".")
    head, tail = whole[:-3], whole[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups + [tail]) + (f".{frac}" if decimals else "")


def write_csv(name: str, rows: list[list], delimiter: str = ",", preamble: str = "", footer: str = "",
              encoding: str = "utf-8", newline: str = "\n") -> None:
    buf = io.StringIO()
    csv.writer(buf, delimiter=delimiter, lineterminator=newline).writerows(rows)
    (OUT / name).write_bytes((preamble + buf.getvalue() + footer).encode(encoding))


def totals(txns):
    return (round(sum(a for _, _, a, t in txns if t == D), 2), round(sum(a for _, _, a, t in txns if t == C), 2))


EXPECTED: dict[str, dict] = {}


def expect(name, txns, kind="bank", same_as=None, **extra):
    debit, credit = totals(txns)
    EXPECTED[name] = {"kind": kind, "count": len(txns), "total_debit": debit, "total_credit": credit,
                      "same_as": same_as, **extra}


BANK_CHECKS = [
    {"contains": "SALARY ACME", "type": "credit", "amount": 85000.0, "date": "2026-11-05"},
    {"contains": "PRIYA SHARMA", "type": "credit", "amount": 1200.0},
    {"contains": "BIGBASKET", "type": "debit", "amount": 2675.5, "date": "2026-11-09"},
    {"contains": "GST ON MIN BAL", "type": "debit", "amount": 72.0, "category": "Fees & charges"},
]
CARD_CHECKS = [
    {"contains": "CROMA", "type": "debit", "amount": 12999.0, "date": "2026-10-29"},
    {"contains": "BBPS PAYMENT", "type": "credit", "amount": 5000.0},
    {"contains": "REFUND AMAZON", "type": "credit", "amount": 1200.0},
    {"contains": "LATE PAYMENT FEE", "type": "debit", "category": "Fees & charges"},
]


# ---------------------------------------------------------------------------
# 1. HDFC-style CSV: account details on top, short years, separate withdrawal/deposit, summary block at the end
# ---------------------------------------------------------------------------
def hdfc_style():
    rows = [["Date", "Narration", "Chq./Ref.No.", "Value Dt", "Withdrawal Amt.", "Deposit Amt.", "Closing Balance"]]
    for (d, desc, amt, typ), bal in zip(BANK, BAL):
        rows.append([d.strftime("%d/%m/%y"), desc, f"0000{402611000000 + d.day * 37:d}", d.strftime("%d/%m/%y"),
                     f"{amt:.2f}" if typ == D else "", f"{amt:.2f}" if typ == C else "", f"{bal:.2f}"])
    preamble = ("SAMPLE BANK LTD,,,,,,\nStatement of account,,,,,,\nAccount Branch : MG ROAD,,,,,,\n"
                "Account No : XXXXXXXX4521,,,,,,\nStatement From : 01/11/2026 To : 14/11/2026,,,,,,\n,,,,,,\n")
    debit, credit = totals(BANK)
    footer = ("\nSTATEMENT SUMMARY :-,,,,,,\nOpening Balance,Dr Count,Cr Count,Debits,Credits,Closing Bal,\n"
              f"{OPENING:.2f},9,3,{debit:.2f},{credit:.2f},{CLOSING:.2f},\n"
              "Generated On: 15/11/2026,,,,,,\nThis is a computer generated statement.,,,,,,\n")
    write_csv("hdfc_style.csv", rows, preamble=preamble, footer=footer)
    expect("hdfc_style.csv", BANK, same_as="bank", closing_balance=CLOSING, opening_balance=OPENING,
           checks=BANK_CHECKS)


# ---------------------------------------------------------------------------
# 2. SBI-style workbook: a Summary sheet, the table starting at row 18 of a second sheet, "1 Nov 2026" text dates,
#    comma-formatted text amounts, a Total row
# ---------------------------------------------------------------------------
def sbi_style():
    wb = Workbook()
    summary = wb.active
    summary.title = "Summary"
    for r in [["Account Name", "Ravi Kumar"], ["Account Number", "XXXXXXX8812"], ["Branch", "Koramangala"],
              ["Statement Period", "01 Nov 2026 to 14 Nov 2026"], ["Opening Balance", indian(OPENING)],
              ["Closing Balance", indian(CLOSING)]]:
        summary.append(r)
    ws = wb.create_sheet("Statement")
    ws.append(["SAMPLE BANK OF INDIA"])
    for _ in range(15):
        ws.append([])
    ws.append(["Account Statement from 1 Nov 2026 to 14 Nov 2026"])
    ws.append(["Txn Date", "Value Date", "Description", "Ref No./Cheque No.", "Debit", "Credit", "Balance"])
    for (d, desc, amt, typ), bal in zip(BANK, BAL):
        day = f"{d.day} {d.strftime('%b %Y')}"
        ws.append([day, day, "TO TRANSFER-" + desc if typ == D else "BY TRANSFER-" + desc, "TRF" + d.strftime("%d%m"),
                   indian(amt) if typ == D else "", indian(amt) if typ == C else "", indian(bal)])
    debit, credit = totals(BANK)
    ws.append(["", "", "Total", "", indian(debit), indian(credit), ""])
    ws.append(["**This is a computer generated statement and does not require a signature."])
    wb.save(OUT / "sbi_style.xlsx")
    expect("sbi_style.xlsx", BANK, same_as="bank", closing_balance=CLOSING, opening_balance=OPENING, checks=BANK_CHECKS)


# ---------------------------------------------------------------------------
# 3. ICICI-style: semicolons (because amounts contain commas), "(INR )" in headers, serial-number column,
#    the header repeated half-way down (page break in the original), legend lines at the end
# ---------------------------------------------------------------------------
def icici_style():
    header = ["S No.", "Value Date", "Transaction Date", "Cheque Number", "Transaction Remarks",
              "Withdrawal Amount (INR )", "Deposit Amount (INR )", "Balance (INR )"]
    rows = [["DETAILED STATEMENT"], ["Transactions List - Ravi Kumar (INR) - XXXXXXXX3390"], [], header]
    for n, ((d, desc, amt, typ), bal) in enumerate(zip(BANK, BAL), 1):
        if n == 7:
            rows.append(header)
        rows.append([str(n), d.strftime("%d-%m-%Y"), d.strftime("%d-%m-%Y"), "-", desc,
                     indian(amt) if typ == D else "0.00", indian(amt) if typ == C else "0.00", indian(bal)])
    rows += [[], ["Legends Used in Account Statement"], ["INF - Internet Fund Transfer"], ["BIL - Bill payment"]]
    write_csv("icici_style.csv", rows, delimiter=";")
    expect("icici_style.csv", BANK, same_as="bank", closing_balance=CLOSING, checks=BANK_CHECKS)


# ---------------------------------------------------------------------------
# 4. Wallet/UPI-app style: date with time, signed amounts with ₹ and +/-, extra Category / Mode / ID columns
# ---------------------------------------------------------------------------
def signed_amount():
    rows = [["Date & Time", "Transaction ID", "Details", "Category", "Amount (₹)", "Mode", "Balance (₹)"]]
    for n, ((d, desc, amt, typ), bal) in enumerate(zip(BANK, BAL)):
        when = datetime(d.year, d.month, d.day, 9 + n % 10, 15, 22).strftime("%Y-%m-%d %H:%M:%S")
        sign = "-" if typ == D else "+"
        rows.append([when, f"T2611{n:06d}88", desc, "Misc", f"{sign}₹{indian(amt)}", "UPI", f"₹{indian(bal)}"])
    write_csv("signed_amount.csv", rows)
    expect("signed_amount.csv", BANK, same_as="bank", closing_balance=CLOSING, checks=BANK_CHECKS)


# ---------------------------------------------------------------------------
# 5. Credit-card export: purchases positive, payments and refunds NEGATIVE (the opposite of a bank file)
# ---------------------------------------------------------------------------
def card_export():
    rows = [["Transaction Date", "Posted Date", "Card No.", "Description", "Category", "Amount"]]
    for d, desc, amt, typ in CARD:
        rows.append([d.strftime("%d-%m-%Y"), d.strftime("%d-%m-%Y"), "XXXX XXXX XXXX 7788", desc, "General",
                     f"{-amt if typ == C else amt:.2f}"])
    write_csv("card_export.csv", rows)
    expect("card_export.csv", CARD, kind="card", same_as="card", checks=CARD_CHECKS)


# ---------------------------------------------------------------------------
# 6. Tab-separated, one Amount column plus a Dr/Cr column, "Nov 01, 2026" dates, balance with " Cr" suffix
# ---------------------------------------------------------------------------
def drcr_column():
    rows = [["Tran Date", "Particulars", "Amount", "Dr/Cr", "Balance"]]
    for (d, desc, amt, typ), bal in zip(BANK, BAL):
        rows.append([d.strftime("%b %d, %Y"), desc, f"{amt:.2f}", "DR" if typ == D else "CR", f"{bal:.2f} Cr"])
    write_csv("drcr_column.tsv", rows, delimiter="\t")
    expect("drcr_column.tsv", BANK, same_as="bank", closing_balance=CLOSING, checks=BANK_CHECKS)


# ---------------------------------------------------------------------------
# 7. Bank PDF with a drawn table: right-aligned amounts, descriptions wrapped onto a second line, two pages with
#    the header repeated, page footers, opening and closing balance outside the table
# ---------------------------------------------------------------------------
COLS_PDF = [("Date", 40, "left"), ("Description", 100, "left"), ("Ref No", 290, "left"),
            ("Withdrawals", 420, "right"), ("Deposits", 490, "right"), ("Balance", 565, "right")]


def _text(page, x, y, s, size=9, align="left"):
    if align == "right":
        x -= fitz.get_text_length(s, fontname="helv", fontsize=size)
    page.insert_text((x, y), s, fontsize=size, fontname="helv")


def _wrap(text, width=26):
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + [cur]


def bank_table_pdf():
    doc = fitz.open()

    def new_page(first):
        page = doc.new_page()
        y = 50
        if first:
            for line in ["SAMPLE BANK - Savings Account Statement (fictional)", "Account holder: Ravi Kumar",
                         "Account No: XXXXXXXX4521", "Statement Period: 01/11/2026 to 14/11/2026",
                         f"Opening Balance: Rs. {indian(OPENING)}"]:
                _text(page, 40, y, line, size=10)
                y += 16
            y += 10
        for label, x, align in COLS_PDF:
            _text(page, x, y, label, size=9, align=align)
        page.draw_line((36, y + 5), (570, y + 5))
        return page, y + 20

    page, y = new_page(True)
    page_no = 1
    for n, ((d, desc, amt, typ), bal) in enumerate(zip(BANK, BAL)):
        if n == 7:  # second page
            _text(page, 280, 800, f"Page {page_no} of 2", size=8)
            page, y = new_page(False)
            page_no += 1
        lines = _wrap(desc)
        _text(page, 40, y, d.strftime("%d/%m/%Y"))
        _text(page, 100, y, lines[0])
        _text(page, 290, y, f"{4030 + n}{d.day:02d}9912")
        _text(page, 420 if typ == D else 490, y, indian(amt), align="right")
        _text(page, 565, y, indian(bal), align="right")
        for extra in lines[1:]:
            y += 12
            _text(page, 100, y, extra)
        page.draw_line((36, y + 5), (570, y + 5), color=(0.8, 0.8, 0.8))
        y += 18
    _text(page, 40, y + 20, f"Closing Balance: Rs. {indian(CLOSING)}", size=10)
    _text(page, 280, 800, f"Page {page_no} of 2", size=8)
    doc.save(str(OUT / "bank_table.pdf"))
    doc.close()
    checks = BANK_CHECKS + [{"contains": "ACME TECHNOLOGIES PVT LTD", "type": "credit", "amount": 85000.0},
                            {"contains": "MG ROAD", "source_has": "page 2"}]
    expect("bank_table.pdf", BANK, same_as="bank", closing_balance=CLOSING, opening_balance=OPENING, checks=checks)


# ---------------------------------------------------------------------------
# 8. Card PDF, different layout: summary labels in one row with the values underneath, "18 Oct 26" dates,
#    a "CR" marker on credits, "Total Outstanding" / "Minimum Due" / "Payment Due Date" wording
# ---------------------------------------------------------------------------
def card_layout_pdf():
    doc = fitz.open()
    page = doc.new_page()
    _text(page, 40, 50, "SAMPLE BANK PLATINUM CREDIT CARD - Monthly Statement (fictional)", size=12)
    _text(page, 40, 70, "Card Number: XXXX XXXX XXXX 7788      Name: Ravi Kumar", size=9)
    _text(page, 40, 86, "Billing Cycle: 16 Oct 2026 - 15 Nov 2026", size=9)
    _text(page, 40, 102, "Available Credit Limit: Rs. 61,550.00", size=9)
    labels = [("Total Outstanding", 40), ("Minimum Due", 170), ("Payment Due Date", 290), ("Credit Limit", 430)]
    values = ["Rs. 38,450.00", "Rs. 1,930.00", "05 Dec 2026", "Rs. 1,00,000.00"]
    for (label, x), value in zip(labels, values):
        _text(page, x, 130, label, size=9)
        _text(page, x, 146, value, size=11)
    y = 185
    _text(page, 40, y, "Date", size=9)
    _text(page, 120, y, "Transaction Details", size=9)
    _text(page, 400, y, "Reward Points", size=9)
    _text(page, 560, y, "Amount (Rs.)", size=9, align="right")
    page.draw_line((36, y + 5), (570, y + 5))
    y += 20
    for d, desc, amt, typ in CARD:
        _text(page, 40, y, d.strftime("%d %b %y"))
        _text(page, 120, y, desc)
        _text(page, 400, y, str(int(amt // 100)) if typ == D else "0")
        _text(page, 540, y, indian(amt), align="right")
        if typ == C:
            _text(page, 544, y, "CR")
        y += 18
    _text(page, 40, y + 20, "Pay the Total Outstanding by the due date to avoid finance charges.", size=8)
    doc.save(str(OUT / "card_layout2.pdf"))
    doc.close()
    expect("card_layout2.pdf", CARD, kind="card", same_as="card", checks=CARD_CHECKS, **CARD_SUMMARY)


# ---------------------------------------------------------------------------
# 9. Workbook with real Excel dates and numbers, Description + Remarks columns, a blank row and a footer total
# ---------------------------------------------------------------------------
def excel_dates():
    wb = Workbook()
    ws = wb.active
    ws.title = "Transactions"
    ws.append(["Posting Date", "Description", "Remarks", "Debit Amount", "Credit Amount", "Running Balance"])
    for (d, desc, amt, typ), bal in zip(BANK, BAL):
        ws.append([datetime(d.year, d.month, d.day), desc, "via mobile app" if typ == D else "",
                   amt if typ == D else None, amt if typ == C else None, bal])
        ws.cell(ws.max_row, 1).number_format = "dd-mmm-yyyy"
        for col in (4, 5, 6):
            ws.cell(ws.max_row, col).number_format = "#,##0.00"
    ws.append([])
    debit, credit = totals(BANK)
    ws.append(["Total", "", "", debit, credit, None])
    wb.save(OUT / "excel_dates.xlsx")
    expect("excel_dates.xlsx", BANK, same_as="bank", closing_balance=CLOSING, checks=BANK_CHECKS)


# ---------------------------------------------------------------------------
# 10. Messy CSV: BOM + Windows line endings, blank lines, quoted commas, a narration broken over two lines inside
#     quotes, "01.11.2026" dates, debits written with a TRAILING minus ("18,000.00-")
# ---------------------------------------------------------------------------
def messy():
    rows = [['Statement for "Ravi Kumar", generated 15.11.2026'], [], ["Date", "Narration", "Amount", "Balance"]]
    for (d, desc, amt, typ), bal in zip(BANK, BAL):
        text = desc
        if "BILL PAYMENT" in desc:
            text = "BESCOM ELECTRICITY, BILL PAYMENT"  # comma inside a quoted field
        if "SALARY" in desc:
            text = "NEFT CR SALARY\nACME TECHNOLOGIES PVT LTD"  # line break inside the field
        rows.append([d.strftime("%d.%m.%Y"), text, f"{indian(amt)}-" if typ == D else indian(amt), indian(bal)])
        if d.day == 8:
            rows.append([])
    write_csv("messy.csv", rows, encoding="utf-8-sig", newline="\r\n")
    expect("messy.csv", BANK, same_as="bank", closing_balance=CLOSING, checks=BANK_CHECKS)


# ---------------------------------------------------------------------------
# 11. No header row at all: columns are guessed from the values
# ---------------------------------------------------------------------------
def no_header():
    rows = []
    for (d, desc, amt, typ), bal in zip(BANK, BAL):
        rows.append([d.strftime("%d/%m/%Y"), desc, f"{amt:.2f}" if typ == D else "", f"{amt:.2f}" if typ == C else "",
                     f"{bal:.2f}"])
    write_csv("no_header.csv", rows)
    expect("no_header.csv", BANK, same_as="bank", closing_balance=CLOSING, checks=BANK_CHECKS, warning=True)


# ---------------------------------------------------------------------------
# 12. ".xls" that is really an HTML table (what several Indian banks' "Download as Excel" button gives you)
# ---------------------------------------------------------------------------
def html_xls():
    cells = "".join(f"<th>{h}</th>" for h in ["Tran Date", "Chq No", "Particulars", "Debit", "Credit", "Balance", "Init. Br"])
    body = []
    for (d, desc, amt, typ), bal in zip(BANK, BAL):
        body.append("<tr>" + "".join(f"<td>{v}</td>" for v in [
            d.strftime("%d-%m-%Y"), "", desc, indian(amt) if typ == D else "", indian(amt) if typ == C else "",
            indian(bal) + "CR", "0512"]) + "</tr>")
    html = ("<html><head><meta charset='utf-8'></head><body>"
            "<table><tr><td colspan='7'>SAMPLE BANK - Account Statement (fictional)</td></tr>"
            "<tr><td>Account No</td><td>XXXXXXXX7711</td></tr></table>"
            f"<table border='1'><tr>{cells}</tr>{''.join(body)}</table>"
            "<p>Unless the constituent notifies the bank immediately of any discrepancy...</p></body></html>")
    (OUT / "html_export.xls").write_text(html, encoding="utf-8")
    expect("html_export.xls", BANK, same_as="bank", closing_balance=CLOSING, checks=BANK_CHECKS)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for make in (hdfc_style, sbi_style, icici_style, signed_amount, card_export, drcr_column, bank_table_pdf,
                 card_layout_pdf, excel_dates, messy, no_header, html_xls):
        make()
    reference = {"bank": [[d.isoformat(), a, t] for d, _, a, t in BANK], "card": [[d.isoformat(), a, t] for d, _, a, t in CARD]}
    (OUT / "expected.json").write_text(json.dumps({"reference": reference, "files": EXPECTED}, indent=2), encoding="utf-8")
    print(f"{len(EXPECTED)} files + expected.json written to {OUT}")
