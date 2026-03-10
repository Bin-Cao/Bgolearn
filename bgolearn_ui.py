#!/usr/bin/env python3
"""Local browser UI for running Bgolearn without writing code."""

from __future__ import annotations

import cgi
import contextlib
import html
import io
import json
import sys
import traceback
import uuid
from http.cookies import SimpleCookie
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DOWNLOAD_CACHE: dict[str, tuple[str, bytes]] = {}
SESSION_CACHE: dict[str, dict[str, Any]] = {}
SESSION_COOKIE_NAME = "bgolearn_session"
UPLOAD_ACCEPT = ".csv,.xlsx,.xls,text/csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.ms-excel"

REGRESSION_MODELS = {
    "default": ("Gaussian Process (default)", None),
    "SVM": ("SVM", "SVM"),
    "RF": ("Random Forest", "RF"),
    "AdaB": ("AdaBoost", "AdaB"),
    "MLP": ("MLP", "MLP"),
}

CLASSIFICATION_MODELS = {
    "GaussianProcess": "Gaussian Process",
    "LogisticRegression": "Logistic Regression",
    "NaiveBayes": "Naive Bayes",
    "SVM": "SVM",
    "RandomForest": "Random Forest",
}

REGRESSION_ACQUISITIONS = {
    "EI": "Expected Improvement (EI)",
    "EI_log": "Log Expected Improvement",
    "EI_plugin": "Expected Improvement with Plugin",
    "Augmented_EI": "Augmented Expected Improvement (AEI)",
    "EQI": "Expected Quantile Improvement (EQI)",
    "Reinterpolation_EI": "Reinterpolation Expected Improvement",
    "UCB": "Upper Confidence Bound (UCB)",
    "PoI": "Probability of Improvement (PoI)",
    "PES": "Predictive Entropy Search (PES)",
    "Knowledge_G": "Knowledge Gradient (KG)",
}

CLASSIFICATION_ACQUISITIONS = {
    "Least_cfd": "Least Confidence",
    "Margin_S": "Margin Sampling",
    "Entropy": "Entropy",
}


def load_bgolearn_module():
    import Bgolearn.BGOsampling as BGOS  # noqa: N811

    return BGOS


def parse_tabular_bytes(raw_bytes: bytes, filename: str = ""):
    import pandas as pd

    suffix = Path(filename).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(io.BytesIO(raw_bytes))

    if suffix and suffix not in {".csv", ".txt"}:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Please upload a CSV, XLSX, or XLS file."
        )

    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(io.StringIO(raw_bytes.decode(encoding)))
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(raw_bytes))


def form_value(form: cgi.FieldStorage, name: str, default: str = "") -> str:
    value = form.getfirst(name, default)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return value


def checked(form: cgi.FieldStorage, name: str) -> bool:
    return form_value(form, name, "").lower() in {"1", "true", "yes", "on"}


def optional_float(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    return float(value)


def optional_int(value: str) -> int | None:
    value = (value or "").strip()
    if not value:
        return None
    return int(value)


def require_upload(form: cgi.FieldStorage, field_name: str) -> tuple[str, bytes]:
    field = form[field_name]
    if not getattr(field, "file", None):
        raise ValueError(f"Please upload a file for '{field_name}'.")
    raw = field.file.read()
    if not raw:
        raise ValueError(f"Uploaded file '{field_name}' is empty.")
    return getattr(field, "filename", "") or "", raw


def optional_upload(form: cgi.FieldStorage, field_name: str) -> tuple[str, bytes] | None:
    if field_name not in form:
        return None
    field = form[field_name]
    if isinstance(field, list):
        field = field[0]
    filename = getattr(field, "filename", "") or ""
    if not filename or not getattr(field, "file", None):
        return None
    raw = field.file.read()
    if not raw:
        raise ValueError(f"Uploaded file '{field_name}' is empty.")
    return filename, raw


def inspect_upload_columns(form: cgi.FieldStorage, field_name: str) -> list[str]:
    filename, raw = require_upload(form, field_name)
    df = parse_tabular_bytes(raw, filename)
    return [str(column) for column in df.columns]


def infer_feature_columns(training_df, virtual_df, target_column: str | None):
    common = [col for col in training_df.columns if col in set(virtual_df.columns)]
    if target_column in common:
        common.remove(target_column)
    return common


def sort_direction(mission: str, acquisition: str, min_search: bool) -> bool:
    if mission == "Classification":
        return True
    if acquisition == "UCB" and min_search:
        return False
    return True


def dataframe_html(df, classes: str = "data-table") -> str:
    return df.to_html(index=False, classes=classes, border=0, justify="left")


def make_download(filename: str, data: bytes) -> str:
    token = uuid.uuid4().hex
    DOWNLOAD_CACHE[token] = (filename, data)
    return token


def escape(text: Any) -> str:
    return html.escape(str(text))


def option_html(value: Any, label: Any, selected_value: Any = "") -> str:
    selected = " selected" if str(value) == str(selected_value) else ""
    return f'<option value="{escape(value)}"{selected}>{escape(label)}</option>'


def render_options(
    options: list[tuple[Any, Any]],
    selected_value: Any = "",
    *,
    include_blank: bool = False,
    blank_label: str = "None",
) -> str:
    parts: list[str] = []
    if include_blank:
        parts.append(option_html("", blank_label, selected_value))
    parts.extend(option_html(value, label, selected_value) for value, label in options)
    return "".join(parts)


def build_cached_file_entry(filename: str, raw: bytes) -> tuple[dict[str, Any], Any]:
    df = parse_tabular_bytes(raw, filename)
    entry = {
        "filename": filename,
        "raw": raw,
        "headers": [str(column) for column in df.columns],
        "rows": len(df),
    }
    return entry, df


def get_uploaded_or_cached_file(
    form: cgi.FieldStorage,
    session_state: dict[str, Any],
    field_name: str,
    label: str,
) -> tuple[str, bytes, Any, str]:
    uploaded = optional_upload(form, field_name)
    if uploaded is not None:
        filename, raw = uploaded
        entry, df = build_cached_file_entry(filename, raw)
        session_state[field_name] = entry
        return filename, raw, df, "uploaded"

    cached = session_state.get(field_name)
    if cached and cached.get("raw"):
        filename = str(cached.get("filename", ""))
        raw = cached["raw"]
        entry, df = build_cached_file_entry(filename, raw)
        session_state[field_name] = entry
        return filename, raw, df, "cached"

    raise ValueError(f"Please upload a {label} file to get started.")


def extract_form_state(form: cgi.FieldStorage | None) -> dict[str, Any]:
    if form is None:
        return {}
    return {
        "mission": form_value(form, "mission", "Regression"),
        "target_column": form_value(form, "target_column"),
        "target_column_manual": form_value(form, "target_column_manual"),
        "feature_columns": [value for value in form.getlist("feature_columns") if value],
        "feature_columns_manual": form_value(form, "feature_columns_manual"),
        "noise_column": form_value(form, "noise_column"),
        "noise_column_manual": form_value(form, "noise_column_manual"),
        "objective_direction": form_value(form, "objective_direction", "min"),
        "acquisition": form_value(form, "acquisition", "EI"),
        "regression_model": form_value(form, "regression_model", "default"),
        "classifier_model": form_value(form, "classifier_model", "GaussianProcess"),
        "opt_num": form_value(form, "opt_num", "1"),
        "seed": form_value(form, "seed", "42"),
        "cv_mode": form_value(form, "cv_mode", "none"),
        "cv_folds": form_value(form, "cv_folds", "5"),
        "noise_fixed": form_value(form, "noise_fixed"),
        "normalize": checked(form, "normalize"),
        "dynamic_w": checked(form, "dynamic_w"),
        "baseline_t": form_value(form, "baseline_t"),
        "alpha": form_value(form, "alpha", "1"),
        "tao": form_value(form, "tao", "0"),
        "beta": form_value(form, "beta", "0.5"),
        "tao_new": form_value(form, "tao_new", "0"),
        "sam_num": form_value(form, "sam_num", "200"),
        "mc_num": form_value(form, "mc_num", "4"),
        "proc_num": form_value(form, "proc_num"),
    }


def render_feature_picker_html(
    headers: list[str],
    selected_features: list[str],
    target_column: str,
) -> str:
    if not headers:
        return '<div class="muted">Upload or reuse a training file to choose feature columns.</div>'

    default_target = target_column if target_column in headers else headers[-1]
    if selected_features:
        chosen = set(selected_features)
    else:
        chosen = {header for idx, header in enumerate(headers) if header != default_target and idx != len(headers) - 1}
        if len(headers) == 1:
            chosen = {headers[0]}

    parts = []
    for header in headers:
        checked_attr = " checked" if header in chosen else ""
        parts.append(
            f'<label class="check"><input type="checkbox" name="feature_columns" value="{escape(header)}"{checked_attr} />{escape(header)}</label>'
        )
    return "".join(parts)


def render_loaded_data_panel(session_state: dict[str, Any]) -> str:
    def summary(title: str, entry: dict[str, Any] | None) -> str:
        if not entry:
            return (
                f'<div class="summary"><span>{escape(title)}</span><strong>Not loaded</strong>'
                '<small>Upload a file once, then rerun with new parameters.</small></div>'
            )
        headers = entry.get("headers") or []
        return (
            f'<div class="summary"><span>{escape(title)}</span><strong>{escape(entry.get("filename", ""))}</strong>'
            f'<small>{escape(entry.get("rows", 0))} rows • {escape(len(headers))} columns</small></div>'
        )

    return (
        '<div class="summary-grid current-data-grid">'
        + summary("Training data", session_state.get("training_file"))
        + summary("Virtual data", session_state.get("virtual_file"))
        + "</div>"
    )


def render_page(
    content: str,
    message: str = "",
    *,
    session_state: dict[str, Any] | None = None,
    form_state: dict[str, Any] | None = None,
) -> bytes:
    session_state = session_state or {}
    page_state: dict[str, Any] = {
        "mission": "Regression",
        "target_column": "",
        "target_column_manual": "",
        "feature_columns": [],
        "feature_columns_manual": "",
        "noise_column": "",
        "noise_column_manual": "",
        "objective_direction": "min",
        "acquisition": "EI",
        "regression_model": "default",
        "classifier_model": "GaussianProcess",
        "opt_num": "1",
        "seed": "42",
        "cv_mode": "none",
        "cv_folds": "5",
        "noise_fixed": "",
        "normalize": True,
        "dynamic_w": False,
        "baseline_t": "",
        "alpha": "1",
        "tao": "0",
        "beta": "0.5",
        "tao_new": "0",
        "sam_num": "200",
        "mc_num": "4",
        "proc_num": "",
    }
    if form_state:
        page_state.update(form_state)

    training_entry = session_state.get("training_file") or {}
    training_headers = [str(header) for header in training_entry.get("headers") or []]
    if training_headers and not page_state["target_column"]:
        page_state["target_column"] = training_headers[-1]

    current_mission = page_state["mission"] if page_state["mission"] in {"Regression", "Classification"} else "Regression"
    regression_models = json.dumps(
        [{"value": key, "label": label} for key, (label, _) in REGRESSION_MODELS.items()]
    )
    classification_models = json.dumps(
        [{"value": key, "label": label} for key, label in CLASSIFICATION_MODELS.items()]
    )
    regression_acq = json.dumps(
        [{"value": key, "label": label} for key, label in REGRESSION_ACQUISITIONS.items()]
    )
    classification_acq = json.dumps(
        [{"value": key, "label": label} for key, label in CLASSIFICATION_ACQUISITIONS.items()]
    )
    acquisition_options = render_options(
        list(
            CLASSIFICATION_ACQUISITIONS.items()
            if current_mission == "Classification"
            else REGRESSION_ACQUISITIONS.items()
        ),
        page_state["acquisition"],
    )
    regression_model_options = render_options(
        [(key, label) for key, (label, _) in REGRESSION_MODELS.items()],
        page_state["regression_model"],
    )
    classification_model_options = render_options(
        list(CLASSIFICATION_MODELS.items()),
        page_state["classifier_model"],
    )
    target_options = (
        render_options([(header, header) for header in training_headers], page_state["target_column"])
        if training_headers
        else option_html("", "Upload or reuse training data to load columns")
    )
    noise_options = render_options(
        [(header, header) for header in training_headers],
        page_state["noise_column"],
        include_blank=True,
        blank_label="None",
    )
    feature_picker_html = render_feature_picker_html(
        training_headers,
        page_state["feature_columns"],
        page_state["target_column"],
    )
    loaded_data_panel = render_loaded_data_panel(session_state)
    has_loaded_training = bool(training_headers)
    column_status_text = (
        f"Using loaded training data: {training_entry.get('filename', '')} ({len(training_headers)} columns). Upload a new file only to replace it."
        if has_loaded_training
        else "Upload the training file to load column options. After the first run, you can leave files blank to reuse current data."
    )
    column_status_class = "status-note success" if has_loaded_training else "status-note"
    mission_options = render_options(
        [("Regression", "Regression"), ("Classification", "Classification")],
        current_mission,
    )
    objective_options = render_options(
        [("min", "Minimize"), ("max", "Maximize")],
        page_state["objective_direction"],
    )
    cv_mode_options = render_options(
        [("none", "None"), ("LOOCV", "LOOCV"), ("kfold", "K-fold")],
        page_state["cv_mode"],
    )
    normalize_checked = " checked" if page_state["normalize"] else ""
    dynamic_w_checked = " checked" if page_state["dynamic_w"] else ""
    regression_hidden = " hidden" if current_mission != "Regression" else ""
    classifier_hidden = " hidden" if current_mission == "Regression" else ""
    direction_hidden = " hidden" if current_mission != "Regression" else ""
    current_training_headers_json = json.dumps(training_headers)
    page = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Bgolearn Studio</title>
  <style>
    :root {{
      --bg: #07111f;
      --panel: rgba(14, 25, 45, 0.82);
      --panel-soft: rgba(255, 255, 255, 0.06);
      --line: rgba(255, 255, 255, 0.12);
      --text: #edf4ff;
      --muted: #9fb0cc;
      --accent: #7dd3fc;
      --accent-2: #a78bfa;
      --success: #86efac;
      --danger: #fda4af;
      --shadow: 0 18px 50px rgba(0, 0, 0, 0.35);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(125, 211, 252, 0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(167, 139, 250, 0.14), transparent 22%),
        linear-gradient(180deg, #08101c, #0b1730 45%, #050b13);
      min-height: 100vh;
    }}
    .shell {{ max-width: 1320px; margin: 0 auto; padding: 32px 20px 60px; }}
    .hero {{
      display: grid; gap: 20px; grid-template-columns: 1.35fr 0.9fr;
      align-items: stretch; margin-bottom: 24px;
    }}
    .card {{
      background: var(--panel); border: 1px solid var(--line); border-radius: 24px;
      box-shadow: var(--shadow); backdrop-filter: blur(18px);
    }}
    .hero-main {{ padding: 28px; }}
    .badge {{
      display: inline-flex; align-items: center; gap: 8px; padding: 8px 14px;
      background: rgba(125, 211, 252, 0.12); border: 1px solid rgba(125, 211, 252, 0.22);
      color: var(--accent); border-radius: 999px; font-size: 13px; font-weight: 600;
    }}
    h1 {{ margin: 16px 0 10px; font-size: 40px; line-height: 1.05; letter-spacing: -0.04em; }}
    h2 {{ margin: 0 0 14px; font-size: 22px; }}
    p {{ color: var(--muted); line-height: 1.6; }}
    .hero-side {{ padding: 24px; display: grid; gap: 14px; }}
    .mini-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .mini {{ background: var(--panel-soft); border: 1px solid var(--line); border-radius: 18px; padding: 16px; }}
    .mini strong {{ display: block; font-size: 24px; margin-bottom: 6px; }}
    .message {{ margin-bottom: 18px; padding: 14px 16px; border-radius: 16px; border: 1px solid var(--line); }}
    .message.error {{ background: rgba(253, 164, 175, 0.09); border-color: rgba(253, 164, 175, 0.35); color: #ffd6dc; }}
    .message.info {{ background: rgba(125, 211, 252, 0.09); border-color: rgba(125, 211, 252, 0.28); color: #d8f2ff; }}
    .layout {{ display: grid; gap: 22px; grid-template-columns: 460px minmax(0, 1fr); }}
    .form-card {{ padding: 24px; position: sticky; top: 18px; align-self: start; }}
    .results-card {{ padding: 24px; min-height: 420px; }}
    .section-title {{ margin: 24px 0 14px; font-size: 14px; letter-spacing: 0.08em; text-transform: uppercase; color: #bdd3f8; }}
    form {{ display: grid; gap: 14px; }}
    .field {{ display: grid; gap: 8px; }}
    .field label {{ font-size: 14px; font-weight: 600; color: #eef5ff; }}
    .field small {{ color: var(--muted); }}
    input[type="text"], input[type="number"], select, textarea {{
      width: 100%; border-radius: 14px; border: 1px solid var(--line); background: rgba(5, 10, 20, 0.45);
      color: var(--text); padding: 12px 14px; font-size: 14px; outline: none;
    }}
    input[type="file"] {{ width: 100%; color: var(--muted); }}
    .grid-2, .grid-3 {{ display: grid; gap: 12px; }}
    .grid-2 {{ grid-template-columns: 1fr 1fr; }}
    .grid-3 {{ grid-template-columns: repeat(3, 1fr); }}
    .switches {{ display: flex; flex-wrap: wrap; gap: 12px; }}
    .check {{ display: inline-flex; align-items: center; gap: 10px; padding: 10px 12px; border-radius: 14px; background: var(--panel-soft); border: 1px solid var(--line); }}
    .picker {{ max-height: 160px; overflow: auto; border: 1px solid var(--line); border-radius: 16px; padding: 12px; background: rgba(5, 10, 20, 0.35); }}
    .picker-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
    .chip {{ font-size: 12px; color: var(--accent); background: rgba(125, 211, 252, 0.1); border: 1px solid rgba(125, 211, 252, 0.24); padding: 4px 10px; border-radius: 999px; display: inline-flex; width: fit-content; }}
    .btn {{
      border: 0; cursor: pointer; border-radius: 16px; padding: 14px 18px; font-weight: 700; font-size: 15px;
      color: #05111d; background: linear-gradient(135deg, #7dd3fc, #a78bfa); box-shadow: 0 12px 30px rgba(125, 211, 252, 0.22);
    }}
    .btn:hover {{ transform: translateY(-1px); }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 18px; }}
    .summary {{ padding: 16px; border-radius: 18px; background: var(--panel-soft); border: 1px solid var(--line); }}
    .summary span {{ display: block; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }}
    .summary strong {{ font-size: 20px; word-break: break-word; }}
    .summary small {{ display: block; margin-top: 8px; color: var(--muted); line-height: 1.5; }}
    .actions {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0 22px; }}
    .link-btn {{ text-decoration: none; color: var(--text); border: 1px solid var(--line); padding: 10px 14px; border-radius: 14px; background: var(--panel-soft); }}
    .table-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 18px; margin-bottom: 18px; }}
    table.data-table {{ width: 100%; border-collapse: collapse; min-width: 680px; }}
    table.data-table th, table.data-table td {{ padding: 12px 14px; border-bottom: 1px solid rgba(255, 255, 255, 0.08); text-align: left; }}
    table.data-table th {{ position: sticky; top: 0; background: rgba(7, 12, 24, 0.96); z-index: 1; font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: #bfd3f4; }}
    code.inline {{ color: #dff6ff; background: rgba(255, 255, 255, 0.08); padding: 2px 8px; border-radius: 999px; }}
    pre {{ background: rgba(5, 10, 20, 0.5); border: 1px solid var(--line); border-radius: 16px; padding: 14px; overflow: auto; color: #dff7ff; }}
    .muted {{ color: var(--muted); }}
    .tip-list {{ margin: 0; padding-left: 18px; color: var(--muted); line-height: 1.6; }}
    .status-note {{ font-size: 12px; color: var(--muted); min-height: 18px; }}
    .status-note.error {{ color: #ffd6dc; }}
    .status-note.success {{ color: #c6f6d5; }}
    .field.inactive {{ opacity: 0.45; }}
    .btn-secondary {{ color: var(--text); background: rgba(255, 255, 255, 0.06); border: 1px solid var(--line); box-shadow: none; }}
    .helper-note {{ margin: 0; font-size: 13px; color: var(--muted); }}
    .current-data-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); margin-bottom: 0; }}
    .hidden {{ display: none; }}
    @media (max-width: 1100px) {{
      .hero, .layout {{ grid-template-columns: 1fr; }}
      .form-card {{ position: static; }}
      .summary-grid {{ grid-template-columns: 1fr 1fr; }}
    }}
    @media (max-width: 700px) {{
      .grid-2, .grid-3, .mini-grid, .summary-grid {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 32px; }}
      .picker-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="card hero-main">
        <div class="badge">Local • Browser-based • No code required</div>
        <h1>Bgolearn Studio</h1>
        <p>Run Bayesian optimization from CSV files in a clean local interface. Upload your training set and virtual design space, choose a built-in Bgolearn method, and download ranked recommendations in seconds.</p>
        <div class="actions">
          <span class="chip">Works with the code in <code class="inline">src/Bgolearn</code></span>
          <span class="chip">Regression and classification</span>
          <span class="chip">CSV in, CSV out</span>
          <a class="link-btn" href="https://bin-cao.github.io/" target="_blank" rel="noopener noreferrer">Created by Bin Cao</a>
        </div>
      </div>
      <div class="card hero-side">
        <h2>What this UI does</h2>
        <div class="mini-grid">
          <div class="mini"><strong>1</strong><p>Upload a training dataset and a virtual sample space.</p></div>
          <div class="mini"><strong>2</strong><p>Select the target, features, model, and acquisition function.</p></div>
          <div class="mini"><strong>3</strong><p>Run Bgolearn directly from your browser.</p></div>
          <div class="mini"><strong>4</strong><p>Download recommendations, full scores, and execution logs.</p></div>
        </div>
      </div>
    </section>

    {message}

    <section class="layout">
      <aside class="card form-card">
        <h2>Run configuration</h2>
        <p class="muted">Tip: your training file can be CSV or Excel. Load data once, then rerun with new parameters without re-uploading unless you want to replace the current files.</p>
        <form method="post" action="/run" enctype="multipart/form-data">
          <div class="section-title">Current loaded data</div>
          {loaded_data_panel}
          <p class="helper-note">Leave the file inputs blank to reuse the current loaded data. Upload new files only when you want to replace them.</p>

          <div class="section-title">Data</div>
          <div class="field">
            <label for="training_file">Training data file</label>
            <input id="training_file" name="training_file" type="file" accept="{UPLOAD_ACCEPT}" />
            <small>Optional on later runs. Choose a file here only if you want to load or replace the current training dataset.</small>
            <div id="column_status" class="{column_status_class}">{escape(column_status_text)}</div>
          </div>
          <div class="field">
            <label for="virtual_file">Virtual samples file</label>
            <input id="virtual_file" name="virtual_file" type="file" accept="{UPLOAD_ACCEPT}" />
            <small>Optional on later runs. Leave blank to reuse the current virtual dataset.</small>
          </div>
          <div class="field">
            <label for="mission">Mission</label>
            <select id="mission" name="mission">{mission_options}</select>
          </div>

          <div class="section-title">Columns</div>
          <div class="field">
            <label>Feature columns</label>
            <div class="picker"><div id="feature_picker" class="picker-grid">{feature_picker_html}</div></div>
            <small>These columns must exist in both uploaded files. CSV headers can be previewed immediately in the browser.</small>
            <input id="feature_columns_manual" name="feature_columns_manual" type="text" value="{escape(page_state['feature_columns_manual'])}" placeholder="Fallback: x1, x2, x3" />
          </div>
          <div class="grid-2">
            <div class="field">
              <label for="target_column">Target column</label>
              <select id="target_column" name="target_column">{target_options}</select>
              <input id="target_column_manual" name="target_column_manual" type="text" value="{escape(page_state['target_column_manual'])}" placeholder="Fallback: type target column name manually" />
            </div>
            <div class="field">
              <label for="noise_column">Noise column (optional)</label>
              <select id="noise_column" name="noise_column">{noise_options}</select>
              <input id="noise_column_manual" name="noise_column_manual" type="text" value="{escape(page_state['noise_column_manual'])}" placeholder="Fallback: type noise column name manually" />
            </div>
          </div>

          <div class="section-title">Core options</div>
          <div class="grid-2">
            <div class="field">
              <label for="acquisition">Acquisition function</label>
              <select id="acquisition" name="acquisition">{acquisition_options}</select>
            </div>
            <div class="field{regression_hidden}" id="regression_model_field">
              <label for="regression_model">Regression model</label>
              <select id="regression_model" name="regression_model">{regression_model_options}</select>
            </div>
          </div>
          <div class="grid-2" id="classification_row">
            <div class="field{classifier_hidden}" id="classifier_model_field">
              <label for="classifier_model">Classifier</label>
              <select id="classifier_model" name="classifier_model">{classification_model_options}</select>
            </div>
            <div class="field{direction_hidden}" id="objective_direction_field">
              <label for="objective_direction">Objective direction</label>
              <select id="objective_direction" name="objective_direction">{objective_options}</select>
            </div>
          </div>
          <div class="grid-3">
            <div class="field">
              <label for="opt_num">Number of recommendations</label>
              <input id="opt_num" name="opt_num" type="number" min="1" value="{escape(page_state['opt_num'])}" />
            </div>
            <div class="field">
              <label for="seed">Random seed</label>
              <input id="seed" name="seed" type="number" value="{escape(page_state['seed'])}" />
            </div>
            <div class="field">
              <label for="cv_mode">Cross-validation</label>
              <select id="cv_mode" name="cv_mode">{cv_mode_options}</select>
            </div>
          </div>
          <div class="grid-2">
            <div class="field">
              <label for="cv_folds">K-fold value</label>
              <input id="cv_folds" name="cv_folds" type="number" min="2" value="{escape(page_state['cv_folds'])}" />
            </div>
            <div class="field">
              <label for="noise_fixed">Fixed noise std (optional)</label>
              <input id="noise_fixed" name="noise_fixed" type="number" step="any" value="{escape(page_state['noise_fixed'])}" placeholder="Example: 0.2" />
            </div>
          </div>
          <div class="switches">
            <label class="check"><input type="checkbox" name="normalize"{normalize_checked} /> Normalize features</label>
            <label class="check"><input type="checkbox" name="dynamic_w"{dynamic_w_checked} /> Dynamic weighting</label>
          </div>

          <div class="section-title">Advanced method parameters</div>
          <div class="grid-3">
            <div class="field" data-param="baseline_t">
              <label for="baseline_t">Baseline T</label>
              <input id="baseline_t" name="baseline_t" type="number" step="any" value="{escape(page_state['baseline_t'])}" placeholder="For EI / EI_log / PoI" />
            </div>
            <div class="field" data-param="alpha">
              <label for="alpha">Alpha</label>
              <input id="alpha" name="alpha" type="number" step="any" value="{escape(page_state['alpha'])}" placeholder="For AEI / UCB" />
            </div>
            <div class="field" data-param="tao">
              <label for="tao">Tao</label>
              <input id="tao" name="tao" type="number" step="any" value="{escape(page_state['tao'])}" placeholder="For AEI / PoI" />
            </div>
          </div>
          <div class="grid-3">
            <div class="field" data-param="beta">
              <label for="beta">Beta</label>
              <input id="beta" name="beta" type="number" step="any" value="{escape(page_state['beta'])}" placeholder="For EQI" />
            </div>
            <div class="field" data-param="tao_new">
              <label for="tao_new">EQI noise</label>
              <input id="tao_new" name="tao_new" type="number" step="any" value="{escape(page_state['tao_new'])}" placeholder="For EQI" />
            </div>
            <div class="field" data-param="sam_num">
              <label for="sam_num">PES samples</label>
              <input id="sam_num" name="sam_num" type="number" min="1" value="{escape(page_state['sam_num'])}" />
            </div>
          </div>
          <div class="grid-2">
            <div class="field" data-param="mc_num">
              <label for="mc_num">Knowledge Gradient MC</label>
              <input id="mc_num" name="mc_num" type="number" min="1" value="{escape(page_state['mc_num'])}" />
            </div>
            <div class="field" data-param="proc_num">
              <label for="proc_num">Knowledge Gradient processes</label>
              <input id="proc_num" name="proc_num" type="number" min="1" value="{escape(page_state['proc_num'])}" placeholder="Blank = single process" />
            </div>
          </div>
          <div id="parameter_hint" class="status-note">Choose an acquisition function to see which advanced parameters are relevant.</div>

          <div class="actions">
            <button class="btn" type="submit">Run Bgolearn</button>
            <button class="btn btn-secondary" type="submit" formaction="/clear-data" formnovalidate>Clear current data</button>
          </div>
        </form>
      </aside>

      <main class="card results-card">
        {content}
      </main>
    </section>
  </div>

  <script>
    const regressionModels = {regression_models};
    const classificationModels = {classification_models};
    const regressionAcq = {regression_acq};
    const classificationAcq = {classification_acq};
    const currentTrainingHeaders = {current_training_headers_json};
    const acquisitionParams = {{
      EI: ['baseline_t'],
      EI_log: ['baseline_t'],
      EI_plugin: [],
      Augmented_EI: ['alpha', 'tao'],
      EQI: ['beta', 'tao_new'],
      Reinterpolation_EI: [],
      UCB: ['alpha'],
      PoI: ['baseline_t', 'tao'],
      PES: ['sam_num'],
      Knowledge_G: ['mc_num', 'proc_num'],
      Least_cfd: [],
      Margin_S: [],
      Entropy: [],
    }};

    function parseCsvHeader(text) {{
      const line = (text.split(/\r?\n/).find(Boolean) || '').trim();
      const values = [];
      let current = '';
      let inQuotes = false;
      for (let i = 0; i < line.length; i += 1) {{
        const ch = line[i];
        if (ch === '"') {{
          if (inQuotes && line[i + 1] === '"') {{ current += '"'; i += 1; }}
          else {{ inQuotes = !inQuotes; }}
        }} else if (ch === ',' && !inQuotes) {{
          values.push(current.trim());
          current = '';
        }} else {{
          current += ch;
        }}
      }}
      values.push(current.trim());
      return values.filter(Boolean);
    }}

    function setOptions(selectEl, options, includeBlank = false) {{
      const existing = selectEl.value;
      selectEl.innerHTML = '';
      if (includeBlank) {{
        const blank = document.createElement('option');
        blank.value = '';
        blank.textContent = 'None';
        selectEl.appendChild(blank);
      }}
      options.forEach((item) => {{
        const option = document.createElement('option');
        option.value = item.value;
        option.textContent = item.label;
        selectEl.appendChild(option);
      }});
      if (options.some((item) => item.value === existing)) {{
        selectEl.value = existing;
      }}
    }}

    function renderFeaturePicker(headers) {{
      const picker = document.getElementById('feature_picker');
      const target = document.getElementById('target_column').value;
      picker.innerHTML = '';
      headers.forEach((header, idx) => {{
        const wrapper = document.createElement('label');
        wrapper.className = 'check';
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.name = 'feature_columns';
        input.value = header;
        input.checked = header !== target && idx !== headers.length - 1;
        if (headers.length === 1) input.checked = true;
        const text = document.createTextNode(header);
        wrapper.appendChild(input);
        wrapper.appendChild(text);
        picker.appendChild(wrapper);
      }});
    }}

    function updateColumnSelectors(headers) {{
      const targetSelect = document.getElementById('target_column');
      const noiseSelect = document.getElementById('noise_column');
      const currentTarget = targetSelect.value;
      setOptions(targetSelect, headers.map((h) => ({{ value: h, label: h }})));
      setOptions(noiseSelect, headers.map((h) => ({{ value: h, label: h }})), true);
      if (headers.includes(currentTarget)) {{
        targetSelect.value = currentTarget;
      }} else if (headers.length) {{
        targetSelect.value = headers[headers.length - 1];
      }}
      renderFeaturePicker(headers);
    }}

    async function loadTrainingColumns(fileInput) {{
      const file = fileInput.files[0];
      const statusEl = document.getElementById('column_status');
      if (!file) {{
        statusEl.textContent = 'Upload the training file to load column options.';
        statusEl.className = 'status-note';
        return;
      }}

      const lowerName = String(file.name || '').toLowerCase();
      if (lowerName.endsWith('.csv') || lowerName.endsWith('.txt')) {{
        const reader = new FileReader();
        reader.onload = (event) => {{
          const headers = parseCsvHeader(String(event.target.result || ''));
          if (headers.length) {{
            updateColumnSelectors(headers);
            statusEl.textContent = 'Loaded ' + headers.length + ' columns from ' + file.name + '.';
            statusEl.className = 'status-note success';
          }}
        }};
        reader.readAsText(file.slice(0, 4096));
        return;
      }}

      statusEl.textContent = 'Reading columns from ' + file.name + '...';
      statusEl.className = 'status-note';
      try {{
        const formData = new FormData();
        formData.append('data_file', file);
        const response = await fetch('/inspect', {{ method: 'POST', body: formData }});
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || 'Failed to read uploaded file.');
        updateColumnSelectors(payload.headers || []);
        statusEl.textContent = 'Loaded ' + payload.headers.length + ' columns from ' + file.name + '.';
        statusEl.className = 'status-note success';
      }} catch (error) {{
        statusEl.textContent = error.message || 'Failed to read uploaded file.';
        statusEl.className = 'status-note error';
      }}
    }}

    function updateParameterFields() {{
      const acquisition = document.getElementById('acquisition').value;
      const active = new Set(acquisitionParams[acquisition] || []);
      const hintEl = document.getElementById('parameter_hint');
      document.querySelectorAll('[data-param]').forEach((field) => {{
        const paramName = field.dataset.param;
        const isActive = active.has(paramName);
        field.classList.toggle('inactive', !isActive);
      }});
      hintEl.textContent = active.size
        ? 'Relevant parameters for ' + acquisition + ': ' + Array.from(active).join(', ') + '.'
        : 'No extra advanced parameters are required for ' + acquisition + '.';
    }}

    function updateMissionFields() {{
      const mission = document.getElementById('mission').value;
      const acquisitionSelect = document.getElementById('acquisition');
      const regressionModelField = document.getElementById('regression_model_field');
      const classifierField = document.getElementById('classifier_model_field');
      const directionField = document.getElementById('objective_direction_field');

      if (mission === 'Regression') {{
        setOptions(acquisitionSelect, regressionAcq);
        setOptions(document.getElementById('regression_model'), regressionModels);
        regressionModelField.classList.remove('hidden');
        classifierField.classList.add('hidden');
        directionField.classList.remove('hidden');
      }} else {{
        setOptions(acquisitionSelect, classificationAcq);
        setOptions(document.getElementById('classifier_model'), classificationModels);
        regressionModelField.classList.add('hidden');
        classifierField.classList.remove('hidden');
        directionField.classList.add('hidden');
      }}
      updateParameterFields();
    }}

    document.getElementById('training_file').addEventListener('change', (e) => loadTrainingColumns(e.target));
    document.getElementById('mission').addEventListener('change', updateMissionFields);
    document.getElementById('acquisition').addEventListener('change', updateParameterFields);
    document.getElementById('target_column').addEventListener('change', () => {{
      const trainingFile = document.getElementById('training_file');
      if (trainingFile.files.length) loadTrainingColumns(trainingFile);
      else if (currentTrainingHeaders.length) updateColumnSelectors(currentTrainingHeaders);
    }});
    updateMissionFields();
  </script>
</body>
</html>
"""
    return page.encode("utf-8")


def default_content() -> str:
    return """
      <h2>Ready to run</h2>
      <p>Upload your CSV or Excel files on the left, choose the target and acquisition function, and press <strong>Run Bgolearn</strong>.</p>
      <div class="summary-grid">
        <div class="summary"><span>Input</span><strong>Training file</strong></div>
        <div class="summary"><span>Search Space</span><strong>Virtual file</strong></div>
        <div class="summary"><span>Output</span><strong>Recommendations</strong></div>
        <div class="summary"><span>Export</span><strong>Downloadable CSV</strong></div>
      </div>
      <h2>Recommended usage</h2>
      <ul class="tip-list">
        <li>For most regression tasks, start with <strong>EI</strong> or <strong>UCB</strong>.</li>
        <li>Use <strong>Minimize</strong> for lower-is-better targets and <strong>Maximize</strong> for higher-is-better targets.</li>
        <li>For classification, choose <strong>Least Confidence</strong>, <strong>Margin Sampling</strong>, or <strong>Entropy</strong>.</li>
        <li>If your training CSV has a measurement-uncertainty column, you can select it as the optional noise column.</li>
      </ul>
    """


def render_results(result: dict[str, Any]) -> str:
    recommendations_table = dataframe_html(result["recommendations_df"])
    preview_table = dataframe_html(result["preview_df"])
    log_text = escape(result["logs"])
    return f"""
      <h2>Run completed</h2>
      <p class="muted">Your Bgolearn run finished successfully. Review the top recommendations below or download the complete result files.</p>
      <div class="summary-grid">
        <div class="summary"><span>Mission</span><strong>{escape(result['mission'])}</strong></div>
        <div class="summary"><span>Acquisition</span><strong>{escape(result['acquisition_label'])}</strong></div>
        <div class="summary"><span>Training rows</span><strong>{escape(result['training_rows'])}</strong></div>
        <div class="summary"><span>Virtual rows</span><strong>{escape(result['virtual_rows'])}</strong></div>
        <div class="summary"><span>Features</span><strong>{escape(', '.join(result['feature_columns']))}</strong></div>
        <div class="summary"><span>Target</span><strong>{escape(result['target_column'])}</strong></div>
        <div class="summary"><span>Model</span><strong>{escape(result['model_label'])}</strong></div>
        <div class="summary"><span>Recommendations</span><strong>{escape(len(result['recommendations_df']))}</strong></div>
      </div>
      <div class="actions">
        <a class="link-btn" href="/download/{escape(result['recommendations_token'])}">Download recommendations CSV</a>
        <a class="link-btn" href="/download/{escape(result['scores_token'])}">Download scored candidates CSV</a>
        <a class="link-btn" href="/download/{escape(result['logs_token'])}">Download execution log</a>
      </div>
      <h2>Recommended candidates</h2>
      <div class="table-wrap">{recommendations_table}</div>
      <h2>Top scored candidates</h2>
      <div class="table-wrap">{preview_table}</div>
      <h2>Execution log</h2>
      <pre>{log_text}</pre>
    """


def build_result(form: cgi.FieldStorage, session_state: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    BGOS = load_bgolearn_module()

    training_name, _, training_df, training_source = get_uploaded_or_cached_file(
        form, session_state, "training_file", "training data"
    )
    virtual_name, _, virtual_df, virtual_source = get_uploaded_or_cached_file(
        form, session_state, "virtual_file", "virtual samples"
    )

    mission = form_value(form, "mission", "Regression")
    target_column = (
        form_value(form, "target_column").strip()
        or form_value(form, "target_column_manual").strip()
        or str(training_df.columns[-1])
    )
    feature_columns = [value for value in form.getlist("feature_columns") if value]
    if not feature_columns:
        manual_features = form_value(form, "feature_columns_manual").strip()
        if manual_features:
            feature_columns = [item.strip() for item in manual_features.split(",") if item.strip()]
    if not feature_columns:
        feature_columns = infer_feature_columns(training_df, virtual_df, target_column)

    if not feature_columns:
        raise ValueError("No feature columns were selected. Please choose at least one feature column.")
    if target_column not in training_df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the training CSV.")

    missing_virtual = [col for col in feature_columns if col not in virtual_df.columns]
    missing_training = [col for col in feature_columns if col not in training_df.columns]
    if missing_training:
        raise ValueError(f"These feature columns are missing from the training CSV: {missing_training}")
    if missing_virtual:
        raise ValueError(f"These feature columns are missing from the virtual CSV: {missing_virtual}")

    x_train = training_df[feature_columns]
    x_virtual = virtual_df[feature_columns]
    if mission == "Regression":
        y = pd.to_numeric(training_df[target_column], errors="raise")
    else:
        y = training_df[target_column]

    objective_direction = form_value(form, "objective_direction", "min")
    min_search = objective_direction != "max"
    opt_num = max(1, int(form_value(form, "opt_num", "1")))
    seed = int(form_value(form, "seed", "42"))
    normalize = checked(form, "normalize")
    dynamic_w = checked(form, "dynamic_w")

    cv_mode = form_value(form, "cv_mode", "none")
    if cv_mode == "none":
        cv_test: Any = False
    elif cv_mode == "LOOCV":
        cv_test = "LOOCV"
    else:
        cv_test = max(2, int(form_value(form, "cv_folds", "5")))

    noise_std: Any = None
    noise_fixed = form_value(form, "noise_fixed")
    noise_column = form_value(form, "noise_column").strip() or form_value(form, "noise_column_manual").strip()
    if noise_column:
        if noise_column not in training_df.columns:
            raise ValueError(f"Noise column '{noise_column}' was not found in the training CSV.")
        noise_std = pd.to_numeric(training_df[noise_column], errors="raise").to_numpy()
    elif noise_fixed.strip():
        noise_std = float(noise_fixed)

    regression_model_key = form_value(form, "regression_model", "default")
    regression_model_label, regression_model = REGRESSION_MODELS.get(
        regression_model_key, REGRESSION_MODELS["default"]
    )
    classifier_key = form_value(form, "classifier_model", "GaussianProcess")

    acquisition = form_value(form, "acquisition", "EI")
    if mission == "Regression" and acquisition not in REGRESSION_ACQUISITIONS:
        acquisition = "EI"
    if mission == "Classification" and acquisition not in CLASSIFICATION_ACQUISITIONS:
        acquisition = "Least_cfd"

    logs_io = io.StringIO()
    with contextlib.redirect_stdout(logs_io), contextlib.redirect_stderr(logs_io):
        bgo = BGOS.Bgolearn()
        fit_kwargs = {
            "data_matrix": x_train,
            "Measured_response": y,
            "virtual_samples": x_virtual,
            "Mission": mission,
            "opt_num": opt_num,
            "seed": seed,
            "Normalize": normalize,
        }
        if mission == "Regression":
            fit_kwargs.update(
                {
                    "noise_std": noise_std,
                    "Kriging_model": regression_model,
                    "min_search": min_search,
                    "CV_test": cv_test,
                    "Dynamic_W": dynamic_w,
                }
            )
        else:
            fit_kwargs.update({"Classifier": classifier_key})

        model = bgo.fit(**fit_kwargs)

        baseline_t = optional_float(form_value(form, "baseline_t"))
        alpha = float(form_value(form, "alpha", "1") or 1)
        tao = float(form_value(form, "tao", "0") or 0)
        beta = float(form_value(form, "beta", "0.5") or 0.5)
        tao_new = float(form_value(form, "tao_new", "0") or 0)
        sam_num = max(1, int(form_value(form, "sam_num", "200") or 200))
        mc_num = max(1, int(form_value(form, "mc_num", "4") or 4))
        proc_num = optional_int(form_value(form, "proc_num"))

        if mission == "Regression":
            method = getattr(model, acquisition)
            if acquisition in {"EI", "EI_log"}:
                scores, candidates = method(T=baseline_t)
            elif acquisition == "Augmented_EI":
                scores, candidates = method(alpha=alpha, tao=tao)
            elif acquisition == "EQI":
                scores, candidates = method(beta=beta, tao_new=tao_new)
            elif acquisition == "UCB":
                scores, candidates = method(alpha=alpha)
            elif acquisition == "PoI":
                scores, candidates = method(tao=tao, T=baseline_t)
            elif acquisition == "PES":
                scores, candidates = method(sam_num=sam_num)
            elif acquisition == "Knowledge_G":
                scores, candidates = method(MC_num=mc_num, Proc_num=proc_num)
            else:
                scores, candidates = method()
        else:
            method = getattr(model, acquisition)
            scores, candidates = method()

    score_series = np.asarray(scores).reshape(-1)
    candidate_array = np.asarray(candidates)
    if candidate_array.ndim == 1:
        candidate_array = candidate_array.reshape(1, -1)

    full_df = virtual_df[feature_columns].copy()
    full_df["score"] = score_series

    model_label = regression_model_label if mission == "Regression" else CLASSIFICATION_MODELS[classifier_key]

    if mission == "Regression":
        full_df["predicted_mean"] = np.asarray(model.virtual_samples_mean).reshape(-1)
        full_df["predicted_std"] = np.asarray(model.virtual_samples_std).reshape(-1)
    else:
        class_labels = [str(item) for item in getattr(model.model, "classes_", [])]
        probs = np.asarray(model.probs)
        if len(class_labels) == probs.shape[1]:
            full_df["predicted_class"] = [class_labels[i] for i in probs.argmax(axis=1)]
        full_df["max_probability"] = probs.max(axis=1)

    recommended_df = pd.DataFrame(candidate_array, columns=feature_columns)
    recommended_df.insert(0, "rank", range(1, len(recommended_df) + 1))

    comparison = full_df[feature_columns].astype(str).agg(" | ".join, axis=1)
    selected_keys = set(recommended_df[feature_columns].astype(str).agg(" | ".join, axis=1))
    full_df["recommended"] = comparison.isin(selected_keys)

    descending = sort_direction(mission, acquisition, min_search)
    preview_df = full_df.sort_values("score", ascending=not descending).head(min(20, len(full_df)))

    recommendations_token = make_download(
        "bgolearn_recommendations.csv", recommended_df.to_csv(index=False).encode("utf-8-sig")
    )
    scores_token = make_download(
        "bgolearn_scored_candidates.csv", full_df.to_csv(index=False).encode("utf-8-sig")
    )
    logs_token = make_download("bgolearn_run.log", logs_io.getvalue().encode("utf-8"))

    acquisition_label = (
        REGRESSION_ACQUISITIONS.get(acquisition)
        if mission == "Regression"
        else CLASSIFICATION_ACQUISITIONS.get(acquisition)
    ) or acquisition

    return {
        "mission": mission,
        "acquisition_label": acquisition_label,
        "training_rows": len(training_df),
        "virtual_rows": len(virtual_df),
        "training_filename": training_name,
        "virtual_filename": virtual_name,
        "training_source": training_source,
        "virtual_source": virtual_source,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "model_label": model_label,
        "recommendations_df": recommended_df,
        "preview_df": preview_df,
        "logs": logs_io.getvalue(),
        "recommendations_token": recommendations_token,
        "scores_token": scores_token,
        "logs_token": logs_token,
    }


def smoke_check() -> int:
    import numpy as np
    import pandas as pd

    BGOS = load_bgolearn_module()
    train = pd.read_csv(ROOT / "Template" / "data.csv")
    x = train[["x"]]
    y = train["y"]
    vs = pd.DataFrame({"x": np.linspace(0.0, 10.0, 41)})
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        model = BGOS.Bgolearn().fit(data_matrix=x, Measured_response=y, virtual_samples=vs, opt_num=2)
        scores, rec = model.EI()
    assert len(scores) == len(vs)
    assert len(rec) == 2
    print("Smoke check passed: Bgolearn UI backend can import the package and execute EI.")
    return 0


class BgolearnUIHandler(BaseHTTPRequestHandler):
    def _get_session_state(self) -> dict[str, Any]:
        cookie_header = self.headers.get("Cookie", "")
        cookies = SimpleCookie()
        if cookie_header:
            cookies.load(cookie_header)

        morsel = cookies.get(SESSION_COOKIE_NAME)
        session_id = morsel.value if morsel else ""
        if not session_id or session_id not in SESSION_CACHE:
            session_id = uuid.uuid4().hex
            SESSION_CACHE[session_id] = {}
            self._session_cookie_value = session_id
        else:
            self._session_cookie_value = None

        self._session_id = session_id
        return SESSION_CACHE[session_id]

    def _maybe_set_session_cookie(self) -> None:
        session_value = getattr(self, "_session_cookie_value", None)
        if not session_value:
            return
        self.send_header(
            "Set-Cookie",
            f"{SESSION_COOKIE_NAME}={session_value}; Path=/; SameSite=Lax; HttpOnly",
        )

    def _send_html(self, body: bytes, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._maybe_set_session_cookie()
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._maybe_set_session_cookie()
        self.end_headers()
        self.wfile.write(body)

    def _parse_form(self) -> cgi.FieldStorage:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" in content_type:
            return cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                },
            )

        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length).decode("utf-8")
        parsed = parse_qs(raw)
        form = cgi.FieldStorage()
        for key, values in parsed.items():
            form.list = form.list or []
            for value in values:
                form.list.append(cgi.MiniFieldStorage(key, value))
        return form

    def _send_download(self, filename: str, data: bytes) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(data)))
        self._maybe_set_session_cookie()
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        session_state = self._get_session_state()
        if parsed.path == "/":
            body = render_page(default_content(), session_state=session_state)
            self._send_html(body)
            return
        if parsed.path.startswith("/download/"):
            token = parsed.path.rsplit("/", 1)[-1]
            payload = DOWNLOAD_CACHE.get(token)
            if not payload:
                self.send_error(HTTPStatus.NOT_FOUND, "Download not found")
                return
            filename, data = payload
            self._send_download(filename, data)
            return
        if parsed.path == "/health":
            self._send_html(b"ok")
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Page not found")

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/run", "/inspect", "/clear-data"}:
            self.send_error(HTTPStatus.NOT_FOUND, "Page not found")
            return
        session_state = self._get_session_state()
        form: cgi.FieldStorage | None = None
        form_state: dict[str, Any] = {}
        try:
            form = self._parse_form()
            form_state = extract_form_state(form)

            if self.path == "/inspect":
                headers = inspect_upload_columns(form, "data_file")
                self._send_json({"headers": headers})
                return

            if self.path == "/clear-data":
                session_state.clear()
                message = '<div class="message info">Current loaded data was cleared. Upload new files to start again.</div>'
                body = render_page(default_content(), message, session_state=session_state, form_state=form_state)
                self._send_html(body)
                return

            result = build_result(form, session_state)
            source_parts = []
            if result["training_source"] == "cached":
                source_parts.append(f"reused training data ({escape(result['training_filename'])})")
            if result["virtual_source"] == "cached":
                source_parts.append(f"reused virtual data ({escape(result['virtual_filename'])})")
            if source_parts:
                source_note = " This run " + " and ".join(source_parts) + "."
            else:
                source_note = " Loaded new data for this run."
            body = render_page(
                render_results(result),
                f'<div class="message info">Run finished successfully.{source_note}</div>',
                session_state=session_state,
                form_state=form_state,
            )
            self._send_html(body)
        except Exception as exc:  # noqa: BLE001
            if self.path == "/inspect":
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            error_detail = escape(traceback.format_exc())
            message = f'<div class="message error"><strong>Run failed.</strong><br>{escape(exc)}</div>'
            content = f"<h2>Unable to complete the run</h2><p>Please check your inputs and try again.</p><pre>{error_detail}</pre>"
            self._send_html(
                render_page(content, message, session_state=session_state, form_state=form_state),
                status=HTTPStatus.BAD_REQUEST,
            )

    def log_message(self, _format: str, *_args: Any) -> None:
        _ = (_format, _args)
        return


def main(argv: list[str]) -> int:
    host = "127.0.0.1"
    port = 8787
    if "--check" in argv:
        return smoke_check()
    if "--host" in argv:
        host = argv[argv.index("--host") + 1]
    if "--port" in argv:
        port = int(argv[argv.index("--port") + 1])

    server = ThreadingHTTPServer((host, port), BgolearnUIHandler)
    print(f"Bgolearn Studio is running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Bgolearn Studio...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))