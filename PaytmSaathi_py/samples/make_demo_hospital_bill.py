"""Generates demo_hospital_bill.pdf - a fictional bill for testing only."""
import pathlib

import pymupdf as fitz


LINE_ITEMS = [
    ("Room rent - single private room (5 days x Rs. 4,000)", 20000),
    ("ICU room charges (1 day x Rs. 8,000)", 8000),
    ("Consultant doctor visits (3 x Rs. 1,500)", 4500),
    ("Surgeon professional fees", 18000),
    ("Operation theatre charges", 9000),
    ("Nursing charges (5 days x Rs. 1,000)", 5000),
    ("Medicines and pharmacy", 7200),
    ("Laboratory investigations and imaging", 4200),
    ("Surgical consumables and dressings", 1800),
    ("Registration charge", 500),
]


def build(path: pathlib.Path) -> None:
    page_width, page_height = 595, 842
    doc = fitz.open()
    page = doc.new_page(width=page_width, height=page_height)

    page.draw_rect(fitz.Rect(36, 34, page_width - 36, 140), color=(0.05, 0.35, 0.55), width=1.5)
    page.insert_text((52, 62), "RIVERBEND COMMUNITY HOSPITAL", fontsize=17, fontname="hebo",
                     color=(0.05, 0.35, 0.55))
    page.insert_text((52, 83), "FICTIONAL DEMO PROVIDER - SAMPLE DOCUMENT ONLY", fontsize=9,
                     fontname="hebo", color=(0.75, 0.15, 0.12))
    page.insert_text((52, 103), "12 Sample Road, Demo City | Not a real hospital", fontsize=9)
    page.insert_text((52, 123), "FINAL HOSPITAL BILL", fontsize=13, fontname="hebo")

    details = [
        ("Bill No.", "RBH-DEMO-2026-1042"),
        ("Patient", "Aarav Sample"),
        ("Age / Gender", "42 / Male"),
        ("Admission", "08/11/2026"),
        ("Discharge", "13/11/2026"),
        ("Diagnosis", "Acute appendicitis"),
        ("Procedure", "Laparoscopic appendectomy"),
        ("Room Type", "Single private room"),
    ]
    left_x, right_x, top_y = 48, 320, 169
    for index, (label, value) in enumerate(details):
        column, row = divmod(index, 4)
        x = left_x if column == 0 else right_x
        y = top_y + row * 22
        page.insert_text((x, y), label + ":", fontsize=9, fontname="hebo")
        page.insert_text((x + 78, y), value, fontsize=9)

    table_top = 278
    page.draw_rect(fitz.Rect(42, table_top - 19, page_width - 42, table_top), color=(0.05, 0.35, 0.55),
                   fill=(0.05, 0.35, 0.55))
    page.insert_text((50, table_top - 6), "Description", fontsize=9, fontname="hebo", color=(1, 1, 1))
    page.insert_text((page_width - 130, table_top - 6), "Amount (Rs.)", fontsize=9,
                     fontname="hebo", color=(1, 1, 1))
    y = table_top + 20
    for index, (description, amount) in enumerate(LINE_ITEMS):
        if index % 2 == 0:
            page.draw_rect(fitz.Rect(42, y - 13, page_width - 42, y + 5), color=None,
                           fill=(0.94, 0.97, 0.98))
        page.insert_text((50, y), description, fontsize=8.5)
        page.insert_text((page_width - 130, y), f"{amount:,.2f}", fontsize=8.5)
        y += 23

    total = sum(amount for _, amount in LINE_ITEMS)
    summary_top = y + 9
    for label, amount, bold in [
        ("Grand Total", total, True),
        ("Advance Paid", 20000, False),
        ("Balance Due", total - 20000, True),
    ]:
        page.draw_line((page_width - 240, summary_top - 12), (page_width - 48, summary_top - 12),
                       color=(0.7, 0.75, 0.78), width=0.5)
        page.insert_text((page_width - 230, summary_top + 2), label, fontsize=9,
                         fontname="hebo" if bold else "helv")
        page.insert_text((page_width - 130, summary_top + 2), f"{amount:,.2f}", fontsize=9,
                         fontname="hebo" if bold else "helv")
        summary_top += 23

    page.insert_text((48, 690), "This fictional sample is for software testing and demonstration only.",
                     fontsize=9, fontname="hebo", color=(0.75, 0.15, 0.12))
    page.insert_text((48, 710), "It is not a real medical record, invoice, or proof of expense.", fontsize=9,
                     color=(0.75, 0.15, 0.12))
    page.insert_text((48, 756), "Authorized Signatory: Demo Billing Desk", fontsize=9)
    page.draw_line((48, 765), (245, 765), color=(0.4, 0.4, 0.4), width=0.6)
    page.insert_text((48, 797), "Please retain this bill with the discharge summary and prescriptions.",
                     fontsize=8, color=(0.35, 0.35, 0.35))

    doc.save(str(path))
    doc.close()


if __name__ == "__main__":
    output = pathlib.Path(__file__).with_name("demo_hospital_bill.pdf")
    build(output)
    print("wrote", output)
