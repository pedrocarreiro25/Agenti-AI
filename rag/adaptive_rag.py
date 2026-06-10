import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RAG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RAG_DIR.parent
DEFAULT_KB_PATH = RAG_DIR / "knowledge_base.json"
DEFAULT_VALIDATED_CSV = PROJECT_ROOT / "data" / "data_processed" / "invoice_structured_fields.csv"

NULL_VALUES = {"", "null", "none", "nan"}

FIELD_NAMES = [
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
]

PROVIDER_ALIASES = {
    "epal": ["epal", "empresa portuguesa das aguas livres"],
    "eamb": ["eamb", "esposende ambiente"],
    "vodafone_pt": ["vodafone portugal", "vodafone port", "vodafone"],
    "vodafone_tr": ["vodafone telekomunikasyon", "vodafone plaza"],
    "galp": ["galp"],
    "edp": ["edp"],
    "unknown": [],
}

DEFAULT_TIPS = {
    "epal": [
        "EPAL invoice numbers often appear near 'FATURA no' or 'Fatura no'.",
        "Payment due dates often appear near 'Data limite de pagamento' or in payment-reference blocks.",
        "Total values may appear as 'Montante' or 'Total' with EUR.",
    ],
    "eamb": [
        "Esposende Ambiente invoices use 'Fatura: FTA ...' for the invoice number.",
        "VAT can appear in the billing summary as an 'IVA' line followed by the amount.",
        "The service/plan is often near 'Tipo de utilizador'.",
    ],
    "vodafone_pt": [
        "Portuguese Vodafone documents often use 'No Documento' for the invoice number.",
        "The billing period is often written as 'Periodo de faturacao: 1 out a 31 out'.",
        "The total is usually close to 'Total da fatura com IVA' or 'Valor deste mes'.",
    ],
    "vodafone_tr": [
        "Turkish Vodafone invoices use 'Fatura ID' for the invoice number.",
        "Invoice date appears near 'Fatura Tarihi'.",
        "Payment due date appears near 'SON ODEME TARIHI'.",
        "Financial totals use TL and often include 'Ara Toplam', 'Devlete Iletilecek Ucretler', and 'FATURA TUTARI'.",
    ],
    "galp": [
        "Galp energy invoices can contain electricity and gas sections; choose invoice type from the dominant service cues.",
        "Consumption values commonly use kWh and appear in consumption detail tables.",
    ],
    "edp": [
        "EDP invoices can include customer and provider VAT numbers near NIF/NIPC labels.",
        "Electricity consumption values commonly use kWh.",
    ],
}

DEFAULT_OCR_CORRECTIONS = {
    "\u00c3\u00a1": "a",
    "\u00c3\u00a2": "a",
    "\u00c3\u00a3": "a",
    "\u00c3\u00a7": "c",
    "\u00c3\u00a9": "e",
    "\u00c3\u00aa": "e",
    "\u00c3\u00ad": "i",
    "\u00c3\u00b3": "o",
    "\u00c3\u00b5": "o",
    "\u00c3\u00ba": "u",
    "\u00e2\u201a\u00ac": "EUR",
    "N\u00c2\u00b0": "No",
    "n.o": "No",
}


@dataclass
class RetrievalResult:
    provider_id: str
    score: float
    knowledge: dict[str, Any]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_text(value: str) -> str:
    value = value or ""
    normalized = unicodedata.normalize("NFKD", value)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return normalized.lower()


def tokenize(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", normalize_text(value)))


def is_null(value: Any) -> bool:
    return str(value or "").strip().lower() in NULL_VALUES


def canonical_provider(value: str, text: str = "") -> str:
    haystack = normalize_text(f"{value} {text}")
    for provider_id, aliases in PROVIDER_ALIASES.items():
        if provider_id == "unknown":
            continue
        if any(alias in haystack for alias in aliases):
            return provider_id
    return "unknown"


def provider_display_name(provider_id: str) -> str:
    names = {
        "epal": "EPAL",
        "eamb": "EAMB - Esposende Ambiente, EM",
        "vodafone_pt": "Vodafone Portugal",
        "vodafone_tr": "VODAFONE TELEKOMUNIKASYON A.S.",
        "galp": "GALP",
        "edp": "EDP",
        "unknown": "Unknown Provider",
    }
    return names.get(provider_id, provider_id)


def empty_kb() -> dict[str, Any]:
    return {
        "version": 1,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "providers": {},
        "global": {
            "common_ocr_corrections": DEFAULT_OCR_CORRECTIONS,
            "validation_history": [],
        },
    }


def load_kb(path: Path = DEFAULT_KB_PATH) -> dict[str, Any]:
    if not path.exists():
        return empty_kb()
    kb = json.loads(path.read_text(encoding="utf-8"))
    ensure_global_defaults(kb)
    return kb


def save_kb(kb: dict[str, Any], path: Path = DEFAULT_KB_PATH) -> None:
    kb["updated_at"] = now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(kb, indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_global_defaults(kb: dict[str, Any]) -> None:
    global_memory = kb.setdefault("global", {})
    corrections = global_memory.setdefault("common_ocr_corrections", {})
    corrections.update(DEFAULT_OCR_CORRECTIONS)
    global_memory.setdefault("validation_history", [])


def ensure_provider(kb: dict[str, Any], provider_id: str) -> dict[str, Any]:
    providers = kb.setdefault("providers", {})
    if provider_id not in providers:
        providers[provider_id] = {
            "provider_id": provider_id,
            "provider_name": provider_display_name(provider_id),
            "aliases": PROVIDER_ALIASES.get(provider_id, []),
            "provider_specific_extraction_tips": DEFAULT_TIPS.get(provider_id, []),
            "common_ocr_corrections": {},
            "known_invoice_layouts": [],
            "previously_validated_invoices": [],
            "human_reviewer_feedback": [],
            "validation_history": [],
        }
    else:
        provider = providers[provider_id]
        tips = provider.setdefault("provider_specific_extraction_tips", [])
        for tip in DEFAULT_TIPS.get(provider_id, []):
            if tip not in tips:
                tips.append(tip)
        provider.setdefault("common_ocr_corrections", {})
        provider.setdefault("known_invoice_layouts", [])
        provider.setdefault("previously_validated_invoices", [])
        provider.setdefault("human_reviewer_feedback", [])
        provider.setdefault("validation_history", [])
    return providers[provider_id]


def row_signature(row: dict[str, str]) -> str:
    source = row.get("source_file") or row.get("ocr_text_file") or ""
    invoice_number = row.get("invoice_number") or ""
    return f"{source}|{invoice_number}"


def non_null_fields(row: dict[str, str]) -> dict[str, str]:
    return {field: row[field] for field in FIELD_NAMES if field in row and not is_null(row[field])}


def summarize_layout(row: dict[str, str]) -> dict[str, Any]:
    labels = []
    if not is_null(row.get("invoice_number")):
        labels.append("invoice_number_present")
    if not is_null(row.get("invoice_date")):
        labels.append("invoice_date_present")
    if not is_null(row.get("payment_due_date")):
        labels.append("payment_due_date_present")
    if not is_null(row.get("total_value")):
        labels.append("total_value_present")
    if not is_null(row.get("total_vat")):
        labels.append("total_vat_present")
    if not is_null(row.get("units_of_consumption")):
        labels.append("consumption_present")

    return {
        "layout_id": row.get("source_file") or row.get("ocr_text_file"),
        "invoice_type": row.get("invoice_type", "null"),
        "field_markers": labels,
        "successful_fields": sorted(non_null_fields(row)),
        "extraction_warnings": row.get("extraction_warnings", "null"),
        "observed_at": now_iso(),
    }


def seed_from_validated_csv(csv_path: Path = DEFAULT_VALIDATED_CSV, kb_path: Path = DEFAULT_KB_PATH) -> dict[str, Any]:
    kb = load_kb(kb_path)
    seen_signatures = {
        item.get("signature")
        for provider in kb.get("providers", {}).values()
        for item in provider.get("previously_validated_invoices", [])
    }
    seen_layouts = {
        (provider_id, layout.get("layout_id"), layout.get("invoice_type"))
        for provider_id, provider in kb.get("providers", {}).items()
        for layout in provider.get("known_invoice_layouts", [])
    }
    seen_validation_events = {
        (provider_id, item.get("source_file"), item.get("valid_invoice"))
        for provider_id, provider in kb.get("providers", {}).items()
        for item in provider.get("validation_history", [])
        if "valid_invoice" in item
    }

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            provider_id = canonical_provider(row.get("provider_name", ""), " ".join(row.values()))
            provider = ensure_provider(kb, provider_id)
            signature = row_signature(row)
            if signature not in seen_signatures:
                provider["previously_validated_invoices"].append(
                    {
                        "signature": signature,
                        "source_file": row.get("source_file", ""),
                        "ocr_text_file": row.get("ocr_text_file", ""),
                        "valid_invoice": row.get("valid_invoice", "null"),
                        "validated_fields": non_null_fields(row),
                        "extraction_warnings": row.get("extraction_warnings", "null"),
                        "added_at": now_iso(),
                    }
                )
                seen_signatures.add(signature)

            layout = summarize_layout(row)
            layout_key = (provider_id, layout.get("layout_id"), layout.get("invoice_type"))
            if layout_key not in seen_layouts:
                provider["known_invoice_layouts"].append(layout)
                seen_layouts.add(layout_key)

            validation_event = {
                "source_file": row.get("source_file", ""),
                "valid_invoice": row.get("valid_invoice", "null"),
                "missing_required_fields": [
                    field for field in FIELD_NAMES if field in row and is_null(row[field])
                ],
                "recorded_at": now_iso(),
            }
            validation_key = (
                provider_id,
                validation_event.get("source_file"),
                validation_event.get("valid_invoice"),
            )
            if validation_key not in seen_validation_events:
                provider["validation_history"].append(validation_event)
                seen_validation_events.add(validation_key)

    save_kb(kb, kb_path)
    return kb


def provider_score(provider: dict[str, Any], query_text: str, provider_hint: str = "", invoice_type: str = "") -> float:
    query_norm = normalize_text(f"{query_text} {provider_hint} {invoice_type}")
    score = 0.0

    aliases = provider.get("aliases", [])
    if provider_hint and canonical_provider(provider_hint) == provider.get("provider_id"):
        score += 5.0
    score += sum(3.0 for alias in aliases if alias and alias in query_norm)
    if invoice_type and invoice_type in query_norm:
        score += 0.5

    query_tokens = tokenize(query_norm)
    provider_tokens = tokenize(
        " ".join(
            [
                provider.get("provider_name", ""),
                " ".join(provider.get("provider_specific_extraction_tips", [])),
                " ".join(
                    " ".join(layout.get("successful_fields", []))
                    for layout in provider.get("known_invoice_layouts", [])
                ),
            ]
        )
    )
    if query_tokens and provider_tokens:
        score += len(query_tokens & provider_tokens) / max(1, len(query_tokens | provider_tokens))
    return round(score, 4)


def retrieve_provider_knowledge(
    query_text: str,
    provider_hint: str = "",
    invoice_type: str = "",
    top_k: int = 3,
    kb_path: Path = DEFAULT_KB_PATH,
) -> list[RetrievalResult]:
    kb = load_kb(kb_path)
    results = []
    for provider_id, provider in kb.get("providers", {}).items():
        score = provider_score(provider, query_text, provider_hint, invoice_type)
        if score > 0:
            results.append(RetrievalResult(provider_id=provider_id, score=score, knowledge=provider))

    return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]


def build_extraction_context(
    query_text: str,
    provider_hint: str = "",
    invoice_type: str = "",
    top_k: int = 2,
    kb_path: Path = DEFAULT_KB_PATH,
) -> dict[str, Any]:
    kb = load_kb(kb_path)
    retrieved = retrieve_provider_knowledge(query_text, provider_hint, invoice_type, top_k, kb_path)
    return {
        "retrieved_at": now_iso(),
        "provider_hint": provider_hint or None,
        "invoice_type_hint": invoice_type or None,
        "common_ocr_corrections": kb.get("global", {}).get("common_ocr_corrections", {}),
        "providers": [
            {
                "provider_id": item.provider_id,
                "score": item.score,
                "provider_name": item.knowledge.get("provider_name"),
                "provider_specific_extraction_tips": item.knowledge.get("provider_specific_extraction_tips", []),
                "common_ocr_corrections": item.knowledge.get("common_ocr_corrections", {}),
                "known_invoice_layouts": item.knowledge.get("known_invoice_layouts", [])[-5:],
                "validated_examples": item.knowledge.get("previously_validated_invoices", [])[-5:],
                "human_reviewer_feedback": item.knowledge.get("human_reviewer_feedback", [])[-5:],
                "validation_history": item.knowledge.get("validation_history", [])[-5:],
            }
            for item in retrieved
        ],
    }


def apply_ocr_corrections(text: str, context: dict[str, Any]) -> str:
    corrected = text
    corrections = dict(context.get("common_ocr_corrections", {}))
    for provider in context.get("providers", []):
        corrections.update(provider.get("common_ocr_corrections", {}))
    for wrong, right in corrections.items():
        corrected = corrected.replace(wrong, right)
    return corrected


def record_feedback(
    provider_hint: str,
    source_file: str,
    field_name: str,
    old_value: str,
    corrected_value: str,
    note: str = "",
    kb_path: Path = DEFAULT_KB_PATH,
) -> dict[str, Any]:
    kb = load_kb(kb_path)
    provider_id = canonical_provider(provider_hint)
    provider = ensure_provider(kb, provider_id)
    feedback = {
        "source_file": source_file,
        "field_name": field_name,
        "old_value": old_value,
        "corrected_value": corrected_value,
        "note": note,
        "recorded_at": now_iso(),
    }
    provider["human_reviewer_feedback"].append(feedback)
    provider["validation_history"].append(
        {
            "source_file": source_file,
            "event": "human_feedback_recorded",
            "field_name": field_name,
            "recorded_at": now_iso(),
        }
    )
    save_kb(kb, kb_path)
    return feedback


def export_context(args: argparse.Namespace) -> None:
    query_text = args.query or ""
    if args.text_file:
        query_text += "\n" + Path(args.text_file).read_text(encoding="utf-8", errors="replace")
    context = build_extraction_context(
        query_text=query_text,
        provider_hint=args.provider or "",
        invoice_type=args.invoice_type or "",
        top_k=args.top_k,
        kb_path=Path(args.kb),
    )
    output = Path(args.output) if args.output else RAG_DIR / "last_retrieval_context.json"
    output.write_text(json.dumps(context, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Retrieval context written to: {output}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive RAG memory for invoice extraction.")
    parser.add_argument("--kb", default=str(DEFAULT_KB_PATH))
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Seed the knowledge base from validated invoice CSV rows.")
    init_parser.add_argument("--csv", default=str(DEFAULT_VALIDATED_CSV))

    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve provider knowledge for a new invoice.")
    retrieve_parser.add_argument("--query", default="")
    retrieve_parser.add_argument("--text-file")
    retrieve_parser.add_argument("--provider", default="")
    retrieve_parser.add_argument("--invoice-type", default="")
    retrieve_parser.add_argument("--top-k", type=int, default=2)
    retrieve_parser.add_argument("--output")

    feedback_parser = subparsers.add_parser("feedback", help="Record human reviewer feedback.")
    feedback_parser.add_argument("--provider", required=True)
    feedback_parser.add_argument("--source-file", required=True)
    feedback_parser.add_argument("--field", required=True)
    feedback_parser.add_argument("--old-value", default="")
    feedback_parser.add_argument("--corrected-value", required=True)
    feedback_parser.add_argument("--note", default="")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    kb_path = Path(args.kb)

    if args.command == "init":
        kb = seed_from_validated_csv(Path(args.csv), kb_path)
        provider_count = len(kb.get("providers", {}))
        invoice_count = sum(
            len(provider.get("previously_validated_invoices", []))
            for provider in kb.get("providers", {}).values()
        )
        print(f"Knowledge base written to: {kb_path}")
        print(f"Providers: {provider_count}")
        print(f"Validated invoice memories: {invoice_count}")
    elif args.command == "retrieve":
        export_context(args)
    elif args.command == "feedback":
        feedback = record_feedback(
            provider_hint=args.provider,
            source_file=args.source_file,
            field_name=args.field,
            old_value=args.old_value,
            corrected_value=args.corrected_value,
            note=args.note,
            kb_path=kb_path,
        )
        print(json.dumps(feedback, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
