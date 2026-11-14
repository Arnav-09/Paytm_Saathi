"""Creates sample evidence files for testing ticket attachments (samples/evidence/).

    python samples/make_evidence_samples.py --name "Your Registered Name" --txn 412345678900

Evidence is accepted only if it belongs to the ticket and the user:
  * SMS text      -> must contain a date/time AND the ticket's transaction ID
  * screenshot/PDF -> must show the ticket's transaction ID AND the user's registered name
Use the SAME transaction ID when you create the ticket, and the name you registered with.
"""

import argparse
import io
import pathlib
import sys

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from ai_engine.fintech import _verhoeff_valid  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "evidence"


def default_txn() -> str:
    # 12 digits like a UPI UTR, chosen so it can't be mistaken for an Aadhaar number.
    return next(str(412345678900 + i) for i in range(10) if not _verhoeff_valid(str(412345678900 + i)))


def image(lines: list[str]) -> Image.Image:
    img = Image.new("RGB", (980, 120 + 58 * len(lines)), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 34)
    except OSError:
        font = ImageFont.load_default()
    d.rectangle([20, 20, 960, img.height - 20], outline=(0, 46, 110), width=4)
    for i, line in enumerate(lines):
        d.text((50, 50 + i * 58), line, fill=(20, 20, 20), font=font)
    return img


def png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def jpg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def receipt(name: str, txn: str) -> list[str]:
    return ["Payment failed - amount debited", "Rs 2,000.00 on 01/11/2026 10:42 AM", f"From: {name.title()}",
            "To: MERCHANT STORE", f"UPI Ref No: {txn}"]


def write(name: str, data: bytes, expect: str) -> None:
    (OUT / name).write_bytes(data)
    print(f"  {expect:<9} {name:<30} {len(data):>7} bytes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="Ravi Kumar", help="the name you registered with")
    ap.add_argument("--txn", default=default_txn(), help="the transaction ID you will enter on the ticket")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for old in OUT.iterdir():
        old.unlink()
    name, txn = args.name, args.txn
    wrong_txn = str(int(txn) + 7) if txn.isdigit() else txn + "X"

    print(f"Ticket transaction ID to enter: {txn}   |   Registered name: {name}\n")
    good = image(receipt(name, txn))
    write("ok_receipt_screenshot.png", png(good), "ACCEPT")
    write("ok_receipt_screenshot.jpg", jpg(good), "ACCEPT")
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in ["Bank statement (sample)", f"Account holder: {name.title()}", "",
                 "Date        Description                     Debit",
                 f"01/11/2026  UPI MERCHANT STORE Ref {txn}   Rs 2,000.00"]:
        page.insert_text((60, y), line, fontsize=12, fontname="helv")
        y += 22
    write("ok_bank_statement.pdf", doc.tobytes(), "ACCEPT")
    doc.close()

    write("bad_wrong_txn_screenshot.png", png(image(receipt(name, wrong_txn))), "REJECT")
    write("bad_wrong_name_screenshot.png", png(image(receipt("Someone Else", txn))), "REJECT")
    write("bad_no_text_screenshot.png", png(Image.new("RGB", (600, 400), (235, 240, 248))), "REJECT")
    write("fake_text_as_png.png", b"I am just a text file pretending to be a PNG image.\n", "REJECT")
    write("fake_exe_as_pdf.pdf", b"MZ\x90\x00\x03\x00\x00\x00 pretend Windows program " + b"\x00" * 64, "REJECT")
    write("fake_jpg_named_png.png", jpg(good), "REJECT")
    write("empty.png", b"", "REJECT")

    sms = OUT / "sms_samples.txt"
    sms.write_text(
        "Paste each SMS into 'Paste SMS text instead'.\n\n"
        "ACCEPT (date + matching transaction ID):\n"
        f"Rs 2,000.00 debited from A/c XX1234 on 01/11/2026 10:42 to MERCHANT STORE. UPI Ref No {txn}. OTP is 482913\n\n"
        "REJECT - no date:\n"
        f"Rs 2,000.00 debited from your account. UPI Ref No {txn}\n\n"
        "REJECT - transaction ID does not match the ticket:\n"
        f"Rs 2,000.00 debited on 01/11/2026. UPI Ref No {wrong_txn}\n\n"
        "REJECT - no transaction ID:\n"
        "Rs 2,000.00 debited from your account on 01/11/2026\n",
        encoding="utf-8")
    print(f"  {'(text)':<9} sms_samples.txt\n\nFolder: {OUT}")


if __name__ == "__main__":
    main()
