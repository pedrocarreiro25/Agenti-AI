import argparse
import csv
import re
import unicodedata
from datetime import datetime
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parent / "data"
DEFAULT_TEXT_DIR = DATA_ROOT / "data_txt"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "data_processed"

NULL_VALUE = "null"

FIELDNAMES = [
    "source_file",
    "ocr_text_file",
    "valid_invoice",
    "invoice_type",
    "invoice_number",
    "invoice_date",
    "currency",
    "payment_due_date",
    "provider_name",
    "provider_vat_number",
    "provider_address",
    "buyer_name",
    "buyer_vat_number",
    "buyer_address",
    "service_plan_name",
    "consumption_start_date",
    "consumption_end_date",
    "units_of_consumption",
    "unit_type",
    "subtotal_value",
    "total_vat",
    "total_value",
    "extraction_warnings",
]

PROVIDER_PATTERNS = [
    ("EPAL", re.compile(r"\bEPAL\b|Empresa Portuguesa\s+das\s+[AÃÁA]guas Livres", re.I)),
    ("EAMB - Esposende Ambiente, EM", re.compile(r"Esposende Ambiente|EAMB", re.I)),
    ("Vodafone Portugal, Comunicações Pessoais S.A.", re.compile(r"Vodafone Port|Vodafone Portugal", re.I)),
    ("VODAFONE TELEKOMUNIKASYON A.S.", re.compile(r"VODAFONE TELEKOMUNIKASYON", re.I)),
    ("GALP", re.compile(r"\bGALP\b|Galp", re.I)),
    ("EDP", re.compile(r"\bEDP\b|eletricidade", re.I)),
]

TYPE_KEYWORDS = [
    ("water", ["agua", "aguas", "epal", "esposende ambiente", "saneamento"]),
    ("natural gas", ["gas natural", "gás natural", "gas"]),
    ("telecom", ["vodafone", "telecom", "gsm", "internet", "gb", "sms", "dk"]),
    ("electricity", ["eletricidade", "electricidade", "kwh", "energia", "potencia"]),
]

DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\b"),
    re.compile(r"\b(\d{4})[./-](\d{1,2})[./-](\d{1,2})\b"),
    re.compile(r"\b(\d{4})(\d{2})(\d{2})\b"),
]

MONEY_RE = re.compile(
    r"(?P<prefix>EUR|TL|€)?\s*"
    r"(?P<value>\d{1,3}(?:[ .]\d{3})*(?:[,.]\d{2,4})|\d{1,6}(?:[,.]\d{2,4}))"
    r"\s*(?P<suffix>EUR|TL|€)?",
    re.I,
)


def fold_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return normalized.lower()


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def null_if_empty(value: str | None) -> str:
    value = normalize_space(value or "")
    return value if value else NULL_VALUE


def read_ocr_body(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    source_match = re.search(r"^SOURCE FILE:\s*(.+)$", text, re.M)
    source_file = normalize_space(source_match.group(1)) if source_match else path.name
    parts = text.split("==============================")
    body = parts[-1] if len(parts) >= 3 else text
    return source_file, body.strip()


def parse_date_parts(parts: tuple[str, ...]) -> str | None:
    if len(parts[0]) == 4:
        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
    else:
        day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
        if year < 100:
            year += 2000 if year < 70 else 1900

    try:
        return datetime(year, month, day).date().isoformat()
    except ValueError:
        return None


def all_dates(text: str) -> list[tuple[str, int]]:
    dates: list[tuple[str, int]] = []
    seen = set()
    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(text):
            parsed = parse_date_parts(match.groups())
            if parsed and parsed not in seen:
                dates.append((parsed, match.start()))
                seen.add(parsed)
    return sorted(dates, key=lambda item: item[1])


def nearest_date(text: str, labels: list[str]) -> str | None:
    folded = fold_text(text)
    dates = all_dates(text)
    best: tuple[int, str] | None = None
    for label in labels:
        for label_match in re.finditer(re.escape(fold_text(label)), folded):
            for date, pos in dates:
                distance = abs(pos - label_match.start())
                if distance < 250 and (best is None or distance < best[0]):
                    best = (distance, date)
    return best[1] if best else None


def normalize_money(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip().replace(" ", "")
    value = re.sub(r"^[^\d]+|[^\d,.]+$", "", value)
    if "." in value and "," in value:
        value = value.replace(".", "").replace(",", ".")
    elif "," in value:
        value = value.replace(",", ".")
    elif re.fullmatch(r"\d{5,}", value):
        value = f"{value[:-2]}.{value[-2:]}"
    try:
        amount = float(value)
    except ValueError:
        return None
    if amount > 100000:
        return None
    return f"{amount:.2f}"


def money_near(text: str, labels: list[str], window: int = 220) -> str | None:
    folded = fold_text(text)
    candidates: list[tuple[int, int, str]] = []
    for label in labels:
        label_folded = fold_text(label)
        for label_match in re.finditer(re.escape(label_folded), folded):
            snippet = text[label_match.start() : label_match.start() + window]
            next_label = re.search(
                r"\n\s*(?:Data|SON ODEME|Resumo|Tarife|Refer|Entidade|Cliente|Taxa|KDV|IVA)\b",
                snippet[1:],
                re.I,
            )
            if next_label:
                snippet = snippet[: next_label.start() + 1]

            for match in MONEY_RE.finditer(snippet):
                raw = match.group(0)
                value = normalize_money(match.group("value"))
                if not value:
                    continue
                before = snippet[max(0, match.start() - 3) : match.start()]
                after = snippet[match.end() : match.end() + 3]
                if "/" in before + after:
                    continue
                has_currency = bool(match.group("prefix") or match.group("suffix") or "€" in raw)
                priority = 0 if has_currency else 1
                candidates.append((priority, match.start(), value))
    if candidates:
        return sorted(candidates, key=lambda item: (item[0], item[1]))[0][2]
    return None


def first_money_in_text(text: str) -> str | None:
    for match in MONEY_RE.finditer(text):
        value = normalize_money(match.group("value"))
        if value:
            return value
    return None


def extract_total_vat(text: str) -> str | None:
    lines = [normalize_space(line) for line in text.splitlines()]
    for index, line in enumerate(lines):
        folded_line = fold_text(line)
        if re.fullmatch(r"iva", folded_line):
            value = first_money_in_text(" ".join(lines[index + 1 : index + 3]))
            if value:
                return value
        if re.search(r"\bKDV\s*%?\s*20\b", line, re.I):
            value = first_money_in_text(line)
            if value:
                return value
        if "devlete" in folded_line and "ucret" in folded_line:
            value = first_money_in_text(" ".join(lines[index : index + 3]))
            if value:
                return value
    return money_near(text, ["Total IVA", "ValorIVA", "KDV %20", "Devlete"])


def extract_invoice_type(stem: str, text: str) -> str | None:
    prefix = stem.split("_", 1)[0].lower()
    if prefix == "agua":
        return "water"
    if prefix == "gas":
        return "natural gas"
    if prefix == "luz":
        return "electricity"
    if prefix == "telecom":
        return "telecom"

    folded = fold_text(text)
    for invoice_type, keywords in TYPE_KEYWORDS:
        if any(keyword in folded for keyword in keywords):
            return invoice_type
    return None


def extract_provider(text: str) -> str | None:
    for provider, pattern in PROVIDER_PATTERNS:
        if pattern.search(text):
            return provider
    lines = [normalize_space(line) for line in text.splitlines() if normalize_space(line)]
    for line in lines[:12]:
        if re.search(r"\b(SA|S\.A\.|EM|A\.S\.)\b", line):
            return line
    return None


def extract_provider_address(text: str) -> str | None:
    lines = [normalize_space(line) for line in text.splitlines()]
    address_bits = []
    for line in lines[:35]:
        if re.search(r"\b(Av\.|Avenida|Rua|Travessa|Cad\.|Caddesi|Plaza|Parque|Sede)\b", line, re.I):
            address_bits.append(line)
        elif address_bits and re.search(r"\b\d{4}-?\d{3}\b|\bIstanbul\b|\bLisboa\b|\bEsposende\b", line, re.I):
            address_bits.append(line)
            break
    return " ".join(address_bits[:3]) or None


def extract_invoice_number(text: str) -> str | None:
    patterns = [
        r"Fatura ID:\s*([A-Z0-9][A-Z0-9./-]+)",
        r"Fatura:\s*([A-Z0-9][A-Z0-9./ -]+)",
        r"FATURA\s*n[oº]?\s*([A-Z0-9./-]+)",
        r"No Documento\s+NoContribuinte\s+N[°o]\s*deConta\s+([A-Z]{1,4}\s*[0-9./-]+)",
        r"\b(FT\s*[A-Z0-9./-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return normalize_space(match.group(1)).rstrip(".")
    return None


def extract_currency(text: str) -> str | None:
    if re.search(r"\bTL\b", text):
        return "TL"
    if re.search(r"€|EUR|\beuros?\b", text, re.I):
        return "EUR"
    return None


def extract_vat_numbers(text: str) -> tuple[str | None, str | None]:
    folded = fold_text(text)
    provider = None
    buyer = None

    provider_match = re.search(r"\b(?:NIPC|NIFC|MPC|NIPC|VD\.?)\D{0,20}((?:PT\s*)?\d[\d\s]{7,14}\d)", text, re.I)
    if provider_match:
        provider = re.sub(r"\D", "", provider_match.group(1))

    buyer_match = re.search(r"\b(?:NoContribuinte|NIF|Contribuinte)\D{0,20}((?:PT\s*)?\d[\d\s]{7,14}\d)", text, re.I)
    if buyer_match:
        buyer = re.sub(r"\D", "", buyer_match.group(1))

    if "vodafone telekomunikasyon" in folded:
        match = re.search(r"\b(\d{3}\s*\d{3}\s*\d{4})\b", text)
        if match:
            provider = re.sub(r"\D", "", match.group(1))

    numbers = [re.sub(r"\D", "", match.group(0)) for match in re.finditer(r"\b(?:PT\s*)?\d{9}\b", text, re.I)]
    numbers = [number for number in numbers if len(number) == 9]
    if provider is None and numbers:
        provider = numbers[-1] if "vodafone port" in folded else numbers[0]
    if buyer is None and len(numbers) > 1:
        buyer = numbers[0] if provider != numbers[0] else numbers[1]
    return provider, buyer


def extract_buyer(text: str) -> tuple[str | None, str | None]:
    lines = [normalize_space(line) for line in text.splitlines() if normalize_space(line)]
    buyer_name = None
    buyer_address = None

    for index, line in enumerate(lines):
        if re.match(r"^(Sr\.|Sn\.)\b", line, re.I):
            buyer_name = normalize_space(re.sub(r"^(Sr\.|Sn\.)\s*", "", line, flags=re.I))
            buyer_address = " ".join(lines[index + 1 : index + 4])
            break

    if buyer_name is None:
        for index, line in enumerate(lines[:55]):
            if re.fullmatch(r"[A-ZÃÁÂÀÇÉÊÍÓÔÕÚÜÑ .'-]{8,}", line) and not re.search(
                r"EPAL|FATURA|VODAFONE|EMPRESA|AMBIENTE|GALP|EDP", line, re.I
            ):
                buyer_name = line
                buyer_address = " ".join(lines[index + 1 : index + 4])
                break

    return buyer_name, buyer_address


def extract_service_plan(text: str) -> str | None:
    candidates = [
        r"\b(Red [A-Za-z0-9 ]{3,60}?(?:Pacote|SMbps|GB))\b",
        r"Tarifa Contratada [^:\n]+:\s*([^\n]+)",
        r"Tipo de utilizador\s+([^\n]+)",
        r"Classe/Tipo Factura[^\n]+-\s*([^\n]+)",
    ]
    for pattern in candidates:
        match = re.search(pattern, text, re.I)
        if match:
            return normalize_space(match.group(1))
    for line in text.splitlines():
        if re.search(r"\b(Agua|Saneamento|CONSUMO DE ELETRICIDADE|Mensalidade)\b", line, re.I):
            return normalize_space(line)
    return None


def extract_period(text: str) -> tuple[str | None, str | None]:
    patterns = [
        r"(\d{4}[./-]\d{2}[./-]\d{2})\s*a\s*(\d{4}[./-]\d{2}[./-]\d{2})",
        r"(\d{1,2}\s+[A-Za-zÃÁÂÀÇÉÊÍÓÔÕÚÜçéêãõ]{3,}\.?)\s*a\s*(\d{1,2}\s+[A-Za-zÃÁÂÀÇÉÊÍÓÔÕÚÜçéêãõ]{3,}\.?)",
        r"\(\s*(\d{1,2}\s+[A-Za-z]{3})\s*-\s*(\d{1,2}\s+[A-Za-z]{3})\s*\)",
    ]
    month_map = {
        "jan": 1,
        "fev": 2,
        "feb": 2,
        "mar": 3,
        "abr": 4,
        "apr": 4,
        "mai": 5,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "ago": 8,
        "aug": 8,
        "set": 9,
        "sep": 9,
        "out": 10,
        "oct": 10,
        "nov": 11,
        "dez": 12,
        "dec": 12,
    }

    invoice_date = nearest_date(text, ["Data de emiss", "Fatura Tarihi", "emitida em"])
    year = int(invoice_date[:4]) if invoice_date else None

    def parse_loose(value: str) -> str | None:
        for pattern in DATE_PATTERNS:
            match = pattern.search(value)
            if match:
                return parse_date_parts(match.groups())
        if year:
            match = re.search(r"(\d{1,2})\s+([A-Za-zÃÁÂÀÇÉÊÍÓÔÕÚÜçéêãõ]{3})", value, re.I)
            if match:
                month = month_map.get(fold_text(match.group(2))[:3])
                if month:
                    try:
                        return datetime(year, month, int(match.group(1))).date().isoformat()
                    except ValueError:
                        return None
        return None

    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return parse_loose(match.group(1)), parse_loose(match.group(2))
    return None, None


def extract_consumption(text: str) -> tuple[str | None, str | None]:
    matches = []
    for match in re.finditer(r"(\d+(?:[,.]\d+)?)\s*(kWh|m3|m\^3|m³|GB|DK|min|SMS)\b", text, re.I):
        value = normalize_money(match.group(1)) or match.group(1).replace(",", ".")
        unit = match.group(2)
        matches.append((value, unit, match.start()))
    if not matches:
        return None, None
    preferred = sorted(matches, key=lambda item: (0 if item[1].lower() in {"kwh", "m3", "m³"} else 1, item[2]))
    return preferred[0][0], preferred[0][1]


def extract_row(path: Path) -> dict[str, str]:
    source_file, text = read_ocr_body(path)
    stem = path.name.replace("_selected_text.txt", "")
    folded = fold_text(text)
    warnings = []

    invoice_type = extract_invoice_type(stem, text)
    invoice_number = extract_invoice_number(text)
    invoice_date = nearest_date(text, ["Data de Emiss", "Data de emiss", "emitida em", "Fatura Tarihi", "Data de emissão"])
    payment_due_date = nearest_date(
        text,
        ["Data limite de pagamento", "DATA LIMITE", "SON ODEME", "débito a partir", "debito a partir"],
    )
    currency = extract_currency(text)
    provider_name = extract_provider(text)
    provider_address = extract_provider_address(text)
    provider_vat, buyer_vat = extract_vat_numbers(text)
    buyer_name, buyer_address = extract_buyer(text)
    service_plan = extract_service_plan(text)
    consumption_start, consumption_end = extract_period(text)
    units, unit_type = extract_consumption(text)

    total_value = money_near(
        text,
        ["Total da Fatura", "Valor da fatura atual", "FATURA TUTARI", "Total fatura", "Montante"],
    )
    subtotal = money_near(text, ["Ara Toplam", "Subtotal", "Valor Base", "Valores sem IVA", "Vergiler Haric"])
    total_vat = extract_total_vat(text)

    valid_signals = [
        bool(invoice_type),
        bool(invoice_number),
        bool(invoice_date),
        bool(total_value),
        "fatura" in folded or "factura" in folded or "invoice" in folded,
    ]
    valid_invoice = sum(valid_signals) >= 2 and len(text) >= 80

    if not valid_invoice:
        warnings.append("OCR text did not contain enough invoice signals.")
    for label, value in [
        ("invoice_number", invoice_number),
        ("invoice_date", invoice_date),
        ("total_value", total_value),
    ]:
        if value is None:
            warnings.append(f"{label} not extracted.")

    row = {
        "source_file": source_file,
        "ocr_text_file": str(path),
        "valid_invoice": str(valid_invoice).lower(),
        "invoice_type": invoice_type,
        "invoice_number": invoice_number,
        "invoice_date": invoice_date,
        "currency": currency,
        "payment_due_date": payment_due_date,
        "provider_name": provider_name,
        "provider_vat_number": provider_vat,
        "provider_address": provider_address,
        "buyer_name": buyer_name,
        "buyer_vat_number": buyer_vat,
        "buyer_address": buyer_address,
        "service_plan_name": service_plan,
        "consumption_start_date": consumption_start,
        "consumption_end_date": consumption_end,
        "units_of_consumption": units,
        "unit_type": unit_type,
        "subtotal_value": subtotal,
        "total_vat": total_vat,
        "total_value": total_value,
        "extraction_warnings": " | ".join(warnings),
    }
    return {key: null_if_empty(row.get(key)) for key in FIELDNAMES}


def extract_batch(text_dir: Path, output_csv: Path) -> list[dict[str, str]]:
    files = sorted(text_dir.glob("*_selected_text.txt"))
    rows = [extract_row(path) for path in files]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract structured invoice fields from selected OCR text files.")
    parser.add_argument("--text-dir", default=str(DEFAULT_TEXT_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "invoice_structured_fields.csv"))
    return parser


def main() -> list[dict[str, str]]:
    args = build_arg_parser().parse_args()
    rows = extract_batch(Path(args.text_dir), Path(args.output))
    print(f"Structured invoice rows: {len(rows)}")
    print(f"Output CSV: {args.output}")
    print(f"Valid invoice rows: {sum(row['valid_invoice'] == 'true' for row in rows)}/{len(rows)}")
    return rows


if __name__ == "__main__":
    main()
