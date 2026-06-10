# Adaptive RAG Memory for Invoice Extraction

This folder contains a local Retrieval-Augmented Generation memory layer for invoice extraction.

The memory is designed to adapt extraction behavior from:

- Previously validated invoices
- Provider-specific extraction tips
- Common OCR corrections
- Known invoice layouts
- Human reviewer feedback
- Validation history

## Files

- `adaptive_rag.py`: local RAG memory and retrieval CLI.
- `knowledge_base.json`: provider memory generated from processed invoice rows.
- `knowledge_base.schema.json`: JSON shape for the memory file.
- `last_retrieval_context.json`: most recent retrieval context sample.

## Initialize Memory

```powershell
py rag\adaptive_rag.py init
```

This seeds `knowledge_base.json` from:

```text
data/data_processed/invoice_structured_fields.csv
```

## Retrieve Context for a New Invoice

```powershell
py rag\adaptive_rag.py retrieve --text-file data\data_txt\telecom_05_selected_text.txt --provider vodafone --invoice-type telecom
```

The output context includes relevant provider tips, OCR corrections, known layouts, validated examples, feedback, and validation history.

## Record Human Feedback

```powershell
py rag\adaptive_rag.py feedback --provider "EPAL" --source-file "agua_02.png" --field "payment_due_date" --old-value "null" --corrected-value "2014-08-14" --note "Due date appears near DATA LIMITE DE PAGAMENTO."
```

New feedback is stored under that provider and becomes retrievable for future invoices.

## How To Use During Extraction

Before extracting fields for a new invoice:

1. Read OCR text.
2. Call `build_extraction_context(...)`.
3. Apply `apply_ocr_corrections(...)` to normalize OCR quirks.
4. Use retrieved provider tips, layouts, and feedback as context for extraction rules or an LLM prompt.
5. After human review, call `record_feedback(...)` so future invoices improve.

