from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse


from extract_invoice_fields import DEFAULT_OUTPUT_DIR, DEFAULT_TEXT_DIR, FIELDNAMES, extract_batch, extract_row, read_ocr_body
from rag.adaptive_rag import (
    FIELD_NAMES as RAG_FIELD_NAMES,
    build_extraction_context,
    canonical_provider,
    ensure_provider,
    load_kb,
    non_null_fields,
    now_iso,
    provider_display_name,
    row_signature,
    save_kb,
    seed_from_validated_csv,
    summarize_layout,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = DEFAULT_OUTPUT_DIR / "invoice_structured_fields.csv"
DEFAULT_KB = PROJECT_ROOT / "rag" / "knowledge_base.json"
NULL_VALUES = {"", "null", "none", "nan"}
AUTO_APPROVAL_MIN_VALIDATED = 5
INVOICE_TYPES = {"electricity", "water", "natural gas", "telecom"}

FIELD_GROUPS = {
    "Core Invoice Information": [
        ("invoice_type", "Invoice Type", "Classification of the invoice"),
        ("invoice_number", "Invoice Number", "Unique invoice identifier issued by the provider"),
        ("invoice_date", "Invoice Date", "Official invoice issuance date"),
        ("currency", "Currency", "Currency used in the invoice"),
        ("payment_due_date", "Payment Due Date", "Deadline for payment, when available"),
    ],
    "Provider Information": [
        ("provider_name", "Provider Name", "Legal company name issuing the invoice"),
        ("provider_vat_number", "Provider VAT Number", "Tax/VAT identification number of the provider"),
        ("provider_address", "Provider Address", "Provider billing or headquarters address, when available"),
    ],
    "Buyer Information": [
        ("buyer_name", "Buyer Name", "Customer or company receiving the invoice"),
        ("buyer_vat_number", "Buyer VAT Number", "Tax/VAT identification number of the customer"),
        ("buyer_address", "Buyer Address", "Customer billing address, when available"),
    ],
    "Billing & Consumption": [
        ("service_plan_name", "Service/Plan Name", "The description in the row"),
        ("consumption_start_date", "Consumption Start Date", "Beginning of the billing or consumption period"),
        ("consumption_end_date", "Consumption End Date", "End of the billing or consumption period"),
        ("units_of_consumption", "Units of Consumption", "Quantity consumed"),
        ("unit_type", "Unit Type", "Unit measurement associated with the consumption value"),
    ],
    "Financial Information": [
        ("subtotal_value", "Subtotal Value", "Amount before taxes"),
        ("total_vat", "Total VAT", "Total tax amount applied to the invoice"),
        ("total_value", "Total Value", "Final payable invoice amount"),
    ],
}

CRITICAL_FIELDS = [
    "invoice_type",
    "invoice_number",
    "invoice_date",
    "currency",
    "provider_name",
    "provider_vat_number",
    "total_value",
]
DATE_FIELDS = {"invoice_date", "payment_due_date", "consumption_start_date", "consumption_end_date"}
MONEY_FIELDS = {"subtotal_value", "total_vat", "total_value"}


@dataclass
class ReviewState:
    provider_id: str
    validated_count: int
    validation_errors: list[str]
    auto_approval: bool
    completion: float


@dataclass
class DashboardConfig:
    csv_path: Path
    kb_path: Path


def e(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def is_null(value: Any) -> bool:
    return str(value or "").strip().lower() in NULL_VALUES


def normalize_cell(value: Any) -> str:
    value = "" if value is None else str(value)
    value = value.strip()
    return value if value else "null"


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append({field: normalize_cell(row.get(field)) for field in FIELDNAMES})
        return rows


def write_rows(csv_path: Path, rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows([{field: normalize_cell(row.get(field)) for field in FIELDNAMES} for row in rows])


def parse_bool(value: Any) -> bool:
    return str(value or "").strip().lower() == "true"


def parse_float(value: Any) -> float | None:
    if is_null(value):
        return None
    try:
        return float(str(value).replace(" ", "").replace(",", "."))
    except ValueError:
        return None


def valid_iso_date(value: Any) -> bool:
    if is_null(value):
        return True
    try:
        datetime.strptime(str(value), "%Y-%m-%d")
    except ValueError:
        return False
    return True


def field_completion(row: dict[str, str]) -> float:
    present = sum(1 for field in RAG_FIELD_NAMES if not is_null(row.get(field)))
    return present / len(RAG_FIELD_NAMES)


def provider_validated_count(kb: dict[str, Any], provider_id: str) -> int:
    provider = kb.get("providers", {}).get(provider_id, {})
    return sum(
        1
        for item in provider.get("previously_validated_invoices", [])
        if str(item.get("valid_invoice", "")).lower() == "true"
    )


def validation_errors(row: dict[str, str]) -> list[str]:
    errors = []
    if not parse_bool(row.get("valid_invoice")):
        errors.append("Invoice is not currently classified as valid.")
    for field in CRITICAL_FIELDS:
        if is_null(row.get(field)):
            errors.append(f"{field} is required for approval.")
    invoice_type = row.get("invoice_type", "")
    if not is_null(invoice_type) and invoice_type not in INVOICE_TYPES:
        errors.append("invoice_type must be electricity, water, natural gas, or telecom.")
    for field in DATE_FIELDS:
        if not valid_iso_date(row.get(field)):
            errors.append(f"{field} must use YYYY-MM-DD.")
    for field in MONEY_FIELDS:
        if not is_null(row.get(field)) and parse_float(row.get(field)) is None:
            errors.append(f"{field} must be numeric.")
    start = row.get("consumption_start_date")
    end = row.get("consumption_end_date")
    if valid_iso_date(start) and valid_iso_date(end) and not is_null(start) and not is_null(end) and start > end:
        errors.append("consumption_start_date cannot be after consumption_end_date.")
    if not is_null(row.get("extraction_warnings")):
        errors.append("Extraction warnings must be resolved before automatic approval.")
    return errors


def review_state(row: dict[str, str], kb: dict[str, Any]) -> ReviewState:
    provider_id = canonical_provider(row.get("provider_name", ""), " ".join(row.values()))
    errors = validation_errors(row)
    count = provider_validated_count(kb, provider_id)
    return ReviewState(
        provider_id=provider_id,
        validated_count=count,
        validation_errors=errors,
        auto_approval=count >= AUTO_APPROVAL_MIN_VALIDATED and not errors,
        completion=field_completion(row),
    )


def status_label(state: ReviewState) -> str:
    if state.auto_approval:
        return "Auto-approved"
    if not state.validation_errors:
        return "Ready for human approval"
    return "Needs review"


def upsert_review_memory(row: dict[str, str], decision: str, note: str, kb_path: Path) -> None:
    kb = load_kb(kb_path)
    provider_id = canonical_provider(row.get("provider_name", ""), " ".join(row.values()))
    provider = ensure_provider(kb, provider_id)
    signature = row_signature(row)
    memory = {
        "signature": signature,
        "source_file": row.get("source_file", ""),
        "ocr_text_file": row.get("ocr_text_file", ""),
        "valid_invoice": row.get("valid_invoice", "null"),
        "validated_fields": non_null_fields(row),
        "extraction_warnings": row.get("extraction_warnings", "null"),
        "review_decision": decision,
        "review_note": note,
        "added_at": now_iso(),
    }
    existing_index = next(
        (
            index
            for index, item in enumerate(provider.get("previously_validated_invoices", []))
            if item.get("signature") == signature
        ),
        None,
    )
    if existing_index is None:
        provider["previously_validated_invoices"].append(memory)
    else:
        provider["previously_validated_invoices"][existing_index].update(memory)

    layout = summarize_layout(row)
    layout_key = (layout.get("layout_id"), layout.get("invoice_type"))
    layouts = provider.setdefault("known_invoice_layouts", [])
    if layout_key not in {(item.get("layout_id"), item.get("invoice_type")) for item in layouts}:
        layouts.append(layout)

    provider.setdefault("validation_history", []).append(
        {
            "source_file": row.get("source_file", ""),
            "event": decision,
            "valid_invoice": row.get("valid_invoice", "null"),
            "missing_required_fields": [field for field in RAG_FIELD_NAMES if is_null(row.get(field))],
            "validation_errors": validation_errors(row),
            "review_note": note,
            "recorded_at": now_iso(),
        }
    )
    save_kb(kb, kb_path)


def record_field_feedback(row: dict[str, str], changes: dict[str, tuple[str, str]], note: str, kb_path: Path) -> None:
    if not changes:
        return
    kb = load_kb(kb_path)
    provider_id = canonical_provider(row.get("provider_name", ""), " ".join(row.values()))
    provider = ensure_provider(kb, provider_id)
    for field_name, (old_value, new_value) in changes.items():
        provider.setdefault("human_reviewer_feedback", []).append(
            {
                "source_file": row.get("source_file", ""),
                "field_name": field_name,
                "old_value": old_value,
                "corrected_value": new_value,
                "note": note,
                "recorded_at": now_iso(),
            }
        )
        provider.setdefault("validation_history", []).append(
            {
                "source_file": row.get("source_file", ""),
                "event": "human_feedback_recorded",
                "field_name": field_name,
                "recorded_at": now_iso(),
            }
        )
    save_kb(kb, kb_path)


def safe_read_text(path_value: str) -> str:
    if is_null(path_value):
        return ""
    path = Path(path_value)
    if not path.exists():
        return ""
    try:
        _, body = read_ocr_body(path)
    except Exception:
        body = path.read_text(encoding="utf-8", errors="replace")
    return body


def provider_summary(kb: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for provider_id, provider in sorted(kb.get("providers", {}).items()):
        output.append(
            {
                "provider_id": provider_id,
                "provider_name": provider.get("provider_name", provider_display_name(provider_id)),
                "validated_invoices": provider_validated_count(kb, provider_id),
                "tips": len(provider.get("provider_specific_extraction_tips", [])),
                "ocr_corrections": len(provider.get("common_ocr_corrections", {})),
                "known_layouts": len(provider.get("known_invoice_layouts", [])),
                "human_feedback": len(provider.get("human_reviewer_feedback", [])),
                "validation_events": len(provider.get("validation_history", [])),
            }
        )
    return output


def query_path(view: str, **params: Any) -> str:
    payload = {"view": view}
    payload.update({key: value for key, value in params.items() if value is not None})
    return "/?" + urlencode(payload)


def table(headers: list[str], rows: list[list[Any]], empty: str = "No rows.") -> str:
    if not rows:
        return f"<p class='muted'>{e(empty)}</p>"
    head = "".join(f"<th>{e(header)}</th>" for header in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def notice(message: str, kind: str = "info") -> str:
    return f"<div class='notice {kind}'>{e(message)}</div>"


def render_layout(title: str, view: str, body: str, config: DashboardConfig, message: str = "") -> bytes:
    nav = [
        ("overview", "Overview"),
        ("review", "Review Queue"),
        ("rag", "RAG Context"),
        ("memory", "Provider Memory"),
        ("schema", "Schema"),
        ("export", "Export"),
    ]
    links = "".join(
        f"<a class='nav-link {'active' if key == view else ''}' href='{query_path(key)}'>{label}</a>"
        for key, label in nav
    )
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{e(title)}</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #647084;
      --line: #dfe5ec;
      --primary: #176b87;
      --primary-strong: #0f4c5c;
      --danger: #9f2331;
      --success: #1f7a4d;
      --warn: #946200;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, Segoe UI, Arial, sans-serif;
      letter-spacing: 0;
    }}
    header {{
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 18px 24px 14px;
      position: sticky;
      top: 0;
      z-index: 5;
    }}
    h1 {{ font-size: 24px; margin: 0 0 4px; }}
    h2 {{ font-size: 18px; margin: 24px 0 12px; }}
    h3 {{ font-size: 15px; margin: 18px 0 8px; }}
    .muted {{ color: var(--muted); }}
    .shell {{ max-width: 1480px; margin: 0 auto; }}
    .nav {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }}
    .nav-link {{
      color: var(--text);
      text-decoration: none;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 12px;
      background: #fbfcfd;
      font-size: 14px;
    }}
    .nav-link.active {{ background: var(--primary); color: white; border-color: var(--primary); }}
    main {{ padding: 22px 24px 48px; }}
    .grid {{ display: grid; gap: 14px; }}
    .metrics {{ grid-template-columns: repeat(5, minmax(140px, 1fr)); }}
    .two {{ grid-template-columns: minmax(0, 1fr) minmax(320px, 0.4fr); align-items: start; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .metric .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
    .metric .value {{ font-size: 28px; font-weight: 700; margin-top: 4px; }}
    .notice {{ border-radius: 8px; padding: 12px 14px; margin: 0 0 16px; border: 1px solid var(--line); }}
    .notice.info {{ background: #edf7fb; border-color: #b8dce9; }}
    .notice.success {{ background: #ebf7ef; border-color: #b7dfc7; }}
    .notice.warning {{ background: #fff7df; border-color: #f0d488; }}
    .notice.danger {{ background: #fff0f1; border-color: #edb7be; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: white; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f2f5f8; color: #314152; font-size: 12px; text-transform: uppercase; }}
    tr:last-child td {{ border-bottom: 0; }}
    .status {{ display: inline-block; border-radius: 999px; padding: 4px 8px; font-size: 12px; font-weight: 650; }}
    .auto {{ color: var(--success); background: #e8f7ef; }}
    .ready {{ color: var(--primary-strong); background: #e8f4f8; }}
    .review {{ color: var(--warn); background: #fff3cd; }}
    a {{ color: var(--primary); }}
    form {{ margin: 0; }}
    fieldset {{ border: 1px solid var(--line); border-radius: 8px; margin: 0 0 14px; padding: 12px; }}
    legend {{ padding: 0 6px; color: #314152; font-weight: 700; }}
    label {{ display: block; font-size: 12px; color: var(--muted); margin-bottom: 5px; }}
    input[type=text], textarea, select {{
      width: 100%;
      border: 1px solid #cdd6df;
      border-radius: 6px;
      padding: 9px 10px;
      min-height: 38px;
      font: inherit;
      background: white;
    }}
    textarea {{ min-height: 82px; resize: vertical; }}
    .field-grid {{ display: grid; grid-template-columns: repeat(2, minmax(220px, 1fr)); gap: 12px; }}
    .actions {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; }}
    button {{
      border: 1px solid var(--primary);
      background: var(--primary);
      color: white;
      border-radius: 6px;
      padding: 9px 13px;
      font: inherit;
      cursor: pointer;
    }}
    button.secondary {{ background: white; color: var(--primary); }}
    button.danger {{ background: var(--danger); border-color: var(--danger); }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #111827;
      color: #f9fafb;
      border-radius: 8px;
      padding: 14px;
      max-height: 460px;
      overflow: auto;
      font-size: 12px;
    }}
    details {{ border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; background: white; margin-bottom: 10px; }}
    summary {{ cursor: pointer; font-weight: 700; }}
    @media (max-width: 900px) {{
      .metrics, .two, .field-grid {{ grid-template-columns: 1fr; }}
      header, main {{ padding-left: 14px; padding-right: 14px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="shell">
      <h1>Invoice Extraction Dashboard</h1>
      <div class="muted">Structured extraction, provider memory, and confidence-based approval workflow.</div>
      <nav class="nav">{links}</nav>
    </div>
  </header>
  <main>
    <div class="shell">
      {notice(message, 'success') if message else ''}
      {body}
      <p class="muted">CSV: {e(config.csv_path)}<br>Knowledge base: {e(config.kb_path)}</p>
    </div>
  </main>
</body>
</html>"""
    return html_doc.encode("utf-8")


def render_metrics(rows: list[dict[str, str]], kb: dict[str, Any]) -> str:
    states = [review_state(row, kb) for row in rows]
    valid_count = sum(1 for row in rows if parse_bool(row.get("valid_invoice")))
    auto_ready = sum(1 for state in states if state.auto_approval)
    review_count = sum(1 for state in states if state.validation_errors)
    avg_completion = round(sum(state.completion for state in states) / len(states) * 100) if states else 0
    metrics = [
        ("Invoices", len(rows)),
        ("Valid invoices", valid_count),
        ("Auto-approval ready", auto_ready),
        ("Needs review", review_count),
        ("Avg. field completion", f"{avg_completion}%"),
    ]
    return "<div class='grid metrics'>" + "".join(
        f"<section class='panel metric'><div class='label'>{e(label)}</div><div class='value'>{e(value)}</div></section>"
        for label, value in metrics
    ) + "</div>"


def row_status_cell(state: ReviewState) -> str:
    label = status_label(state)
    class_name = "auto" if state.auto_approval else "ready" if not state.validation_errors else "review"
    return f"<span class='status {class_name}'>{e(label)}</span>"


def render_overview(rows: list[dict[str, str]], kb: dict[str, Any]) -> str:
    body = [render_metrics(rows, kb), "<h2>Processing Queue</h2>"]
    table_rows = []
    for index, row in enumerate(rows):
        state = review_state(row, kb)
        table_rows.append(
            [
                f"<a href='{query_path('review', invoice=index)}'>{e(Path(row.get('source_file', '')).name)}</a>",
                row_status_cell(state),
                e(row.get("provider_name", "null")),
                e(row.get("invoice_type", "null")),
                e(row.get("invoice_number", "null")),
                e(row.get("invoice_date", "null")),
                e(row.get("total_value", "null")),
                e(state.validated_count),
                e(f"{round(state.completion * 100)}%"),
                e(len(state.validation_errors)),
            ]
        )
    body.append(
        table(
            [
                "Source",
                "Status",
                "Provider",
                "Type",
                "Invoice",
                "Date",
                "Total",
                "Provider memory",
                "Completion",
                "Errors",
            ],
            table_rows,
        )
    )
    body.append(render_pipeline_forms())
    return "".join(body)


def render_pipeline_forms() -> str:
    return """<h2>Pipeline Controls</h2>
<div class="panel actions">
  <form method="post" action="/action"><input type="hidden" name="action" value="batch_extract"><button>Run batch extraction</button></form>
  <form method="post" action="/action"><input type="hidden" name="action" value="seed_memory"><button class="secondary">Seed / refresh RAG memory</button></form>
</div>"""


def invoice_selector(rows: list[dict[str, str]], selected_index: int, view: str) -> str:
    options = []
    for index, row in enumerate(rows):
        label = f"{Path(row.get('source_file', '')).name} | {row.get('provider_name', 'null')} | {row.get('invoice_number', 'null')}"
        selected = " selected" if index == selected_index else ""
        options.append(f"<option value='{index}'{selected}>{e(label)}</option>")
    return f"""<form method="get" action="/" class="panel">
  <input type="hidden" name="view" value="{e(view)}">
  <label for="invoice">Invoice</label>
  <select id="invoice" name="invoice" onchange="this.form.submit()">{''.join(options)}</select>
</form>"""


def render_review(rows: list[dict[str, str]], kb: dict[str, Any], selected_index: int) -> str:
    if not rows:
        return notice("No invoices are available.", "info")
    selected_index = max(0, min(selected_index, len(rows) - 1))
    row = rows[selected_index]
    state = review_state(row, kb)
    body = [invoice_selector(rows, selected_index, "review")]
    kind = "success" if state.auto_approval else "warning" if state.validation_errors else "info"
    body.append(notice(f"{status_label(state)}. Provider has {state.validated_count} validated invoices.", kind))
    if state.validation_errors:
        body.append("<section class='panel'><h2>Validation Errors</h2><ul>")
        body.extend(f"<li>{e(error)}</li>" for error in state.validation_errors)
        body.append("</ul></section>")

    fields_html = []
    for group_name, fields in FIELD_GROUPS.items():
        fields_html.append(f"<fieldset><legend>{e(group_name)}</legend><div class='field-grid'>")
        for field_name, label, description in fields:
            fields_html.append(
                f"<div><label title='{e(description)}' for='{e(field_name)}'>{e(label)}</label>"
                f"<input id='{e(field_name)}' name='{e(field_name)}' type='text' value='{e(row.get(field_name, 'null'))}'></div>"
            )
        fields_html.append("</div></fieldset>")
    checked = " checked" if parse_bool(row.get("valid_invoice")) else ""
    body.append(
        f"""<form method="post" action="/action" class="panel">
  <input type="hidden" name="action" value="save_review">
  <input type="hidden" name="invoice" value="{selected_index}">
  {''.join(fields_html)}
  <fieldset>
    <legend>Review Decision</legend>
    <label><input type="checkbox" name="valid_invoice" value="true"{checked}> Valid invoice</label>
    <label for="extraction_warnings">Extraction warnings</label>
    <textarea id="extraction_warnings" name="extraction_warnings">{e('' if is_null(row.get('extraction_warnings')) else row.get('extraction_warnings', ''))}</textarea>
    <label for="note">Reviewer note</label>
    <textarea id="note" name="note"></textarea>
  </fieldset>
  <div class="actions">
    <button>Save corrections to memory</button>
  </div>
</form>"""
    )
    body.append(
        f"""<div class="panel actions">
  <form method="post" action="/action"><input type="hidden" name="action" value="approve"><input type="hidden" name="invoice" value="{selected_index}"><button>Approve invoice</button></form>
  <form method="post" action="/action"><input type="hidden" name="action" value="reject"><input type="hidden" name="invoice" value="{selected_index}"><button class="danger">Reject invoice</button></form>
  <form method="post" action="/action"><input type="hidden" name="action" value="re_extract"><input type="hidden" name="invoice" value="{selected_index}"><button class="secondary">Re-extract selected OCR</button></form>
</div>"""
    )
    return "".join(body)


def render_rag(rows: list[dict[str, str]], kb: dict[str, Any], selected_index: int, kb_path: Path) -> str:
    if not rows:
        return notice("No invoices are available.", "info")
    selected_index = max(0, min(selected_index, len(rows) - 1))
    row = rows[selected_index]
    state = review_state(row, kb)
    text = safe_read_text(row.get("ocr_text_file", ""))
    body = [invoice_selector(rows, selected_index, "rag")]
    body.append(
        f"<div class='grid metrics'><section class='panel metric'><div class='label'>Provider validated invoices</div><div class='value'>{state.validated_count}</div></section>"
        f"<section class='panel metric'><div class='label'>Auto-approval rule</div><div class='value'>{'Pass' if state.auto_approval else 'Human review'}</div></section></div>"
    )
    if not text:
        body.append(notice("No OCR text file is available for this invoice.", "info"))
        return "".join(body)

    context = build_extraction_context(
        query_text=text,
        provider_hint=row.get("provider_name", ""),
        invoice_type=row.get("invoice_type", ""),
        top_k=2,
        kb_path=kb_path,
    )
    body.append(f"<h2>Retrieved Provider Context</h2><p class='muted'>Retrieved at {e(context.get('retrieved_at'))}</p>")
    for provider in context.get("providers", []):
        section = [f"<details open><summary>{e(provider.get('provider_name'))} - score {e(provider.get('score'))}</summary>"]
        tips = provider.get("provider_specific_extraction_tips", [])
        if tips:
            section.append("<h3>Provider extraction tips</h3><ul>")
            section.extend(f"<li>{e(tip)}</li>" for tip in tips)
            section.append("</ul>")
        section.append("<h3>Recent known layouts</h3>")
        section.append(json_table(provider.get("known_invoice_layouts", [])))
        section.append("<h3>Recent human feedback</h3>")
        section.append(json_table(provider.get("human_reviewer_feedback", [])))
        section.append("<h3>Recent validation history</h3>")
        section.append(json_table(provider.get("validation_history", [])))
        section.append("</details>")
        body.append("".join(section))
    body.append(f"<h2>OCR Text Preview</h2><pre>{e(text[:12000])}</pre>")
    return "".join(body)


def json_table(items: list[dict[str, Any]]) -> str:
    if not items:
        return "<p class='muted'>No entries.</p>"
    headers = sorted({key for item in items for key in item.keys()})
    rows = []
    for item in items:
        rows.append([e(json.dumps(item.get(header, ""), ensure_ascii=False)) for header in headers])
    return table(headers, rows)


def render_memory(kb: dict[str, Any], selected_provider: str = "") -> str:
    summary = provider_summary(kb)
    if not selected_provider and summary:
        selected_provider = summary[0]["provider_id"]
    body = ["<h2>Provider-Specific Memory</h2>"]
    body.append(
        table(
            ["Provider", "Name", "Validated", "Tips", "OCR Corrections", "Layouts", "Feedback", "Validation Events"],
            [
                [
                    f"<a href='{query_path('memory', provider=row['provider_id'])}'>{e(row['provider_id'])}</a>",
                    e(row["provider_name"]),
                    e(row["validated_invoices"]),
                    e(row["tips"]),
                    e(row["ocr_corrections"]),
                    e(row["known_layouts"]),
                    e(row["human_feedback"]),
                    e(row["validation_events"]),
                ]
                for row in summary
            ],
        )
    )
    provider = kb.get("providers", {}).get(selected_provider)
    if not provider:
        return "".join(body)

    body.append(f"<h2>{e(provider.get('provider_name', selected_provider))}</h2>")
    body.append("<div class='grid two'><section class='panel'><h3>Tips</h3><ul>")
    body.extend(f"<li>{e(tip)}</li>" for tip in provider.get("provider_specific_extraction_tips", []))
    body.append(
        f"""</ul><form method="post" action="/action">
  <input type="hidden" name="action" value="add_tip">
  <input type="hidden" name="provider" value="{e(selected_provider)}">
  <label for="tip">Add extraction tip</label>
  <input id="tip" name="tip" type="text">
  <div class="actions"><button>Add tip</button></div>
</form></section><section class='panel'><h3>OCR Corrections</h3>"""
    )
    body.append("<pre>" + e(json.dumps(provider.get("common_ocr_corrections", {}), indent=2, ensure_ascii=False)) + "</pre>")
    body.append(
        f"""<form method="post" action="/action">
  <input type="hidden" name="action" value="add_correction">
  <input type="hidden" name="provider" value="{e(selected_provider)}">
  <div class="field-grid"><div><label for="wrong">OCR text</label><input id="wrong" name="wrong" type="text"></div>
  <div><label for="right">Replacement</label><input id="right" name="right" type="text"></div></div>
  <div class="actions"><button>Add correction</button></div>
</form></section></div>"""
    )
    body.append("<h2>Known Layouts</h2>" + json_table(provider.get("known_invoice_layouts", [])))
    body.append("<h2>Human Feedback</h2>" + json_table(provider.get("human_reviewer_feedback", [])))
    body.append("<h2>Validation History</h2>" + json_table(provider.get("validation_history", [])))
    return "".join(body)


def render_schema() -> str:
    body = ["<h2>Required Extraction Schema</h2>"]
    for group_name, fields in FIELD_GROUPS.items():
        body.append(f"<details open><summary>{e(group_name)}</summary>")
        body.append(
            table(
                ["Field", "Key", "Description", "Approval"],
                [
                    [
                        e(label),
                        f"<code>{e(field_name)}</code>",
                        e(description),
                        e("Approval-critical" if field_name in CRITICAL_FIELDS else "Captured when available"),
                    ]
                    for field_name, label, description in fields
                ],
            )
        )
        body.append("</details>")
    return "".join(body)


def render_export(config: DashboardConfig) -> str:
    kb_exists = config.kb_path.exists()
    return f"""<h2>Machine-readable Outputs</h2>
<div class="panel actions">
  <a class="nav-link" href="/download?file=csv">Download structured CSV</a>
  {'<a class="nav-link" href="/download?file=kb">Download RAG knowledge base</a>' if kb_exists else ''}
</div>"""


def render_page(view: str, query: dict[str, list[str]], config: DashboardConfig, message: str = "") -> bytes:
    rows = read_rows(config.csv_path)
    kb = load_kb(config.kb_path)
    if not rows and view != "schema":
        body = notice("No structured invoice CSV was found. Run batch extraction from Overview.", "info")
        body += render_pipeline_forms()
        return render_layout("Invoice Extraction Dashboard", view, body, config, message)

    selected_index = int(query.get("invoice", ["0"])[0] or 0)
    if view == "review":
        body = render_review(rows, kb, selected_index)
    elif view == "rag":
        body = render_rag(rows, kb, selected_index, config.kb_path)
    elif view == "memory":
        body = render_memory(kb, query.get("provider", [""])[0])
    elif view == "schema":
        body = render_schema()
    elif view == "export":
        body = render_export(config)
    else:
        body = render_overview(rows, kb)
    return render_layout("Invoice Extraction Dashboard", view, body, config, message)


def first_value(form: dict[str, list[str]], key: str, default: str = "") -> str:
    return form.get(key, [default])[0]


def handle_action(form: dict[str, list[str]], config: DashboardConfig) -> tuple[str, str]:
    action = first_value(form, "action")
    rows = read_rows(config.csv_path)
    invoice_index = int(first_value(form, "invoice", "0") or 0)

    if action == "batch_extract":
        extracted = extract_batch(DEFAULT_TEXT_DIR, config.csv_path)
        seed_from_validated_csv(config.csv_path, config.kb_path)
        return query_path("overview"), f"Extracted {len(extracted)} invoice rows."

    if action == "seed_memory":
        seed_from_validated_csv(config.csv_path, config.kb_path)
        return query_path("memory"), "Knowledge base refreshed from structured CSV."

    if action in {"save_review", "approve", "reject", "re_extract"}:
        if not rows:
            return query_path("overview"), "No invoice rows are available."
        invoice_index = max(0, min(invoice_index, len(rows) - 1))
        row = rows[invoice_index]

    if action == "save_review":
        updated_row = dict(row)
        for field in RAG_FIELD_NAMES:
            updated_row[field] = normalize_cell(first_value(form, field, row.get(field, "null")))
        updated_row["valid_invoice"] = "true" if first_value(form, "valid_invoice") == "true" else "false"
        updated_row["extraction_warnings"] = normalize_cell(first_value(form, "extraction_warnings"))
        note = first_value(form, "note")
        changes = {
            field: (row.get(field, "null"), updated_row.get(field, "null"))
            for field in RAG_FIELD_NAMES + ["valid_invoice", "extraction_warnings"]
            if normalize_cell(row.get(field)) != normalize_cell(updated_row.get(field))
        }
        rows[invoice_index] = updated_row
        write_rows(config.csv_path, rows)
        record_field_feedback(updated_row, changes, note, config.kb_path)
        upsert_review_memory(updated_row, "human_review_saved", note, config.kb_path)
        return query_path("review", invoice=invoice_index), f"Saved {len(changes)} correction(s) and updated provider memory."

    if action == "approve":
        rows[invoice_index]["valid_invoice"] = "true"
        rows[invoice_index]["extraction_warnings"] = "null"
        write_rows(config.csv_path, rows)
        upsert_review_memory(rows[invoice_index], "human_approved", "Approved from dashboard.", config.kb_path)
        return query_path("review", invoice=invoice_index), "Invoice approved and stored in provider memory."

    if action == "reject":
        rows[invoice_index]["valid_invoice"] = "false"
        write_rows(config.csv_path, rows)
        upsert_review_memory(rows[invoice_index], "human_rejected", "Rejected from dashboard.", config.kb_path)
        return query_path("review", invoice=invoice_index), "Invoice rejected and validation history updated."

    if action == "re_extract":
        ocr_path = Path(row.get("ocr_text_file", ""))
        if not ocr_path.exists():
            return query_path("review", invoice=invoice_index), "OCR text file does not exist."
        rows[invoice_index] = extract_row(ocr_path)
        write_rows(config.csv_path, rows)
        upsert_review_memory(rows[invoice_index], "re_extracted", "Re-extracted from dashboard.", config.kb_path)
        return query_path("review", invoice=invoice_index), "Selected invoice was re-extracted."

    if action == "add_tip":
        provider_id = first_value(form, "provider", "unknown")
        tip = first_value(form, "tip").strip()
        if tip:
            kb = load_kb(config.kb_path)
            provider = ensure_provider(kb, provider_id)
            provider.setdefault("provider_specific_extraction_tips", []).append(tip)
            save_kb(kb, config.kb_path)
        return query_path("memory", provider=provider_id), "Provider tip saved."

    if action == "add_correction":
        provider_id = first_value(form, "provider", "unknown")
        wrong = first_value(form, "wrong").strip()
        right = first_value(form, "right")
        if wrong:
            kb = load_kb(config.kb_path)
            provider = ensure_provider(kb, provider_id)
            provider.setdefault("common_ocr_corrections", {})[wrong] = right
            save_kb(kb, config.kb_path)
        return query_path("memory", provider=provider_id), "OCR correction saved."

    return query_path("overview"), "No action was applied."


class DashboardHandler(BaseHTTPRequestHandler):
    config: DashboardConfig

    def send_html(self, payload: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def redirect(self, path: str, message: str = "") -> None:
        if message:
            separator = "&" if "?" in path else "?"
            path = f"{path}{separator}{urlencode({'message': message})}"
        self.send_response(303)
        self.send_header("Location", path)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/download":
            self.send_download(query)
            return
        view = query.get("view", ["overview"])[0]
        message = query.get("message", [""])[0]
        payload = render_page(view, query, self.config, message)
        self.send_html(payload)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        form = parse_qs(raw)
        path, message = handle_action(form, self.config)
        self.redirect(path, message)

    def send_download(self, query: dict[str, list[str]]) -> None:
        file_kind = query.get("file", ["csv"])[0]
        path = self.config.kb_path if file_kind == "kb" else self.config.csv_path
        if not path.exists():
            self.send_error(404, "File not found")
            return
        payload = path.read_bytes()
        content_type = "application/json" if path.suffix == ".json" else "text/csv"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Disposition", f"attachment; filename={path.name}")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        return


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the invoice extraction dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--kb", default=str(DEFAULT_KB))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    DashboardHandler.config = DashboardConfig(csv_path=Path(args.csv), kb_path=Path(args.kb))
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
