"""Creates sample statements for the Money Clarity and Loan Pre-Flight demos (samples/statements/).

    python samples/make_statement_samples.py

Files (all fictional data):
  bank_statement_oct_nov_2026.csv / .xlsx  - salary on the 7th, rent, EMI, Netflix price rise, overlapping
                                             music subscriptions, a duplicate Swiggy charge, bank fees, a large
                                             Croma purchase, low balance -> cash crunch before next salary
  credit_card_oct_2026.pdf / credit_card_nov_2026.pdf - two card statements; November's bill is higher
                                             (new purchase, late fee, finance charges + GST, duplicate charge)
  loan_rejection_letter.pdf                - a sample rejection letter for the reason decoder
"""

import pathlib

import fitz  # PyMuPDF
import pandas as pd

OUT = pathlib.Path(__file__).resolve().parent / "statements"

BANK = [  # (dd/mm/yyyy, narration, withdrawal, deposit)
    ("03/10/2026", "UPI/RENT OCT/RAMESH OWNER/402711554321", 18000, 0),
    ("05/10/2026", "NACH DR BAJAJ FINSERV LTD EMI 4XX88", 12500, 0),
    ("07/10/2026", "NEFT CR SALARY ACME TECHNOLOGIES PVT LTD", 0, 85000),
    ("07/10/2026", "POS NETFLIX.COM MUMBAI", 499, 0),
    ("09/10/2026", "UPI/SWIGGY/402811223344/Food order", 386, 0),
    ("10/10/2026", "UPI/SPOTIFY INDIA/402833445566", 119, 0),
    ("12/10/2026", "APPLE MUSIC SUBSCRIPTION", 99, 0),
    ("14/10/2026", "UPI/BIGBASKET/402855667788/Groceries", 2340, 0),
    ("15/10/2026", "BESCOM ELECTRICITY BILL PAYMENT", 1840, 0),
    ("18/10/2026", "UPI/UBER INDIA/402877889900", 412, 0),
    ("20/10/2026", "LIC OF INDIA PREMIUM POLICY 88XX21", 4200, 0),
    ("22/10/2026", "UPI/ZOMATO/402899001122", 524, 0),
    ("26/10/2026", "AMAZON PAY INDIA ORDER", 1899, 0),
    ("28/10/2026", "SMS CHARGES QTR", 15, 0),
    ("28/10/2026", "GST ON SMS CHARGES", 2.70, 0),
    ("31/10/2026", "UPI/SWIGGY/402900112233/Food order", 612, 0),
    ("03/11/2026", "UPI/RENT NOV/RAMESH OWNER/403011554321", 18000, 0),
    ("05/11/2026", "NACH DR BAJAJ FINSERV LTD EMI 4XX88", 12500, 0),
    ("07/11/2026", "NEFT CR SALARY ACME TECHNOLOGIES PVT LTD", 0, 85000),
    ("07/11/2026", "POS NETFLIX.COM MUMBAI", 649, 0),
    ("08/11/2026", "UPI/SWIGGY/403011223344/Food order", 456, 0),
    ("08/11/2026", "UPI/SWIGGY/403011223399/Food order", 456, 0),
    ("09/11/2026", "UPI/CROMA RETAIL/403022334455/Laptop", 54990, 0),
    ("10/11/2026", "UPI/SPOTIFY INDIA/403033445566", 119, 0),
    ("10/11/2026", "UPI/BIGBASKET/403055667788/Groceries", 2675, 0),
    ("11/11/2026", "MIN BAL CHARGES OCT", 400, 0),
    ("11/11/2026", "GST ON MIN BAL CHARGES", 72, 0),
    ("12/11/2026", "APPLE MUSIC SUBSCRIPTION", 99, 0),
    ("12/11/2026", "UPI/UBER INDIA/403077889900", 388, 0),
    ("13/11/2026", "UPI/ZOMATO/403099001122", 698, 0),
]
OPENING = 9_500.00  # low on purpose: the demo should show a cash crunch before the next salary

CARD = {
    "credit_card_oct_2026.pdf": {
        "period": "16/09/2026 to 15/10/2026", "due": "05/11/2026", "total": "Rs 21,326.00", "min": "Rs 1,070.00",
        "lines": [
            ("18/09/2026", "PAYMENT RECEIVED - THANK YOU", "18,450.00", "Cr"),
            ("20/09/2026", "AMAZON SELLER SERVICES", "6,499.00", "Dr"),
            ("24/09/2026", "ZOMATO", "642.00", "Dr"),
            ("27/09/2026", "INDIAN OIL FUEL STATION", "3,200.00", "Dr"),
            ("27/09/2026", "FUEL SURCHARGE", "32.00", "Dr"),
            ("02/10/2026", "MYNTRA DESIGNS", "4,350.00", "Dr"),
            ("06/10/2026", "SWIGGY", "580.00", "Dr"),
            ("09/10/2026", "BIGBASKET", "2,890.00", "Dr"),
            ("12/10/2026", "UBER INDIA", "433.00", "Dr"),
            ("14/10/2026", "AMAZON SELLER SERVICES", "2,700.00", "Dr"),
        ],
    },
    "credit_card_nov_2026.pdf": {
        "period": "16/10/2026 to 15/11/2026", "due": "05/12/2026", "total": "Rs 38,450.00", "min": "Rs 1,930.00",
        "lines": [
            ("08/11/2026", "PAYMENT RECEIVED - THANK YOU", "5,000.00", "Cr"),
            ("18/10/2026", "AMAZON SELLER SERVICES", "3,999.00", "Dr"),
            ("21/10/2026", "ZOMATO", "389.00", "Dr"),
            ("21/10/2026", "ZOMATO", "389.00", "Dr"),
            ("25/10/2026", "INDIAN OIL FUEL STATION", "3,400.00", "Dr"),
            ("29/10/2026", "CROMA RETAIL ELECTRONICS", "12,999.00", "Dr"),
            ("02/11/2026", "SWIGGY", "720.00", "Dr"),
            ("06/11/2026", "LATE PAYMENT FEE", "1,000.00", "Dr"),
            ("06/11/2026", "GST ON LATE PAYMENT FEE", "180.00", "Dr"),
            ("09/11/2026", "BIGBASKET", "3,150.00", "Dr"),
            ("12/11/2026", "FINANCE CHARGES", "1,250.00", "Dr"),
            ("12/11/2026", "GST ON FINANCE CHARGES", "225.00", "Dr"),
            ("14/11/2026", "MYNTRA DESIGNS", "2,450.00", "Dr"),
        ],
    },
}


def bank_files() -> None:
    rows, balance = [], OPENING
    for d, narration, wd, dep in BANK:
        balance = round(balance - wd + dep, 2)
        rows.append({"Date": d, "Narration": narration, "Chq./Ref.No.": "", "Value Dt": d,
                     "Withdrawal Amt.": f"{wd:.2f}" if wd else "", "Deposit Amt.": f"{dep:.2f}" if dep else "",
                     "Closing Balance": f"{balance:.2f}"})
    frame = pd.DataFrame(rows)
    preamble = ("SAMPLE BANK - Account Statement (fictional)\nAccount holder: Ravi Kumar\n"
                "Account No: XXXXXXXX4521\nStatement from 01/10/2026 to 13/11/2026\n\n")
    (OUT / "bank_statement_oct_nov_2026.csv").write_text(preamble + frame.to_csv(index=False), encoding="utf-8")
    frame.to_excel(OUT / "bank_statement_oct_nov_2026.xlsx", index=False)
    print(f"  bank statement: {len(rows)} rows, closing balance Rs {balance:,.2f}")


def card_pdf(name: str, spec: dict) -> None:
    doc = fitz.open()
    page = doc.new_page()
    lines = ["SAMPLE BANK CREDIT CARD STATEMENT (fictional)", "Card: XXXX XXXX XXXX 7788   Name: Ravi Kumar",
             f"Statement Period: {spec['period']}", f"Payment Due Date: {spec['due']}",
             f"Total Amount Due: {spec['total']}", f"Minimum Amount Due: {spec['min']}",
             "Credit Limit: Rs 1,00,000.00", "", "Date        Transaction Details                     Amount"]
    lines += [f"{d} {desc} {amt} {drcr}" for d, desc, amt, drcr in spec["lines"]]
    y = 60
    for i, line in enumerate(lines):
        page.insert_text((40, y), line, fontsize=13 if i == 0 else 10, fontname="helv")
        y += 20
    doc.save(str(OUT / name))
    doc.close()
    print(f"  {name}: {len(spec['lines'])} transactions")


def rejection_letter() -> None:
    doc = fitz.open()
    page = doc.new_page()
    text = [
        "SAMPLE FINANCE LTD (fictional) - Loan Application Update", "", "Date: 10/11/2026", "Dear Applicant,", "",
        "Thank you for applying for a personal loan of Rs 5,00,000. After review, we are unable to approve",
        "your application at this time for the following reasons:", "",
        "1. Your existing obligations are high relative to your income (FOIR above our policy limit).",
        "2. Multiple enquiries were found on your credit report in the last 6 months.",
        "3. Your credit score is below our minimum requirement for unsecured loans.", "",
        "You may reapply after 6 months. This decision does not affect your existing relationship with us.",
    ]
    y = 60
    for line in text:
        page.insert_text((40, y), line, fontsize=11, fontname="helv")
        y += 20
    doc.save(str(OUT / "loan_rejection_letter.pdf"))
    doc.close()
    print("  loan_rejection_letter.pdf")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    bank_files()
    for name, spec in CARD.items():
        card_pdf(name, spec)
    rejection_letter()
    print(f"\nFolder: {OUT}")
