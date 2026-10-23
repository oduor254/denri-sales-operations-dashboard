"""Customer feedback, read from the same Google Sheet the POS Request Hub uses.

The Request Hub looks a customer up in this sheet to decide whether they
qualify for the feedback discount; the sheet itself never reaches Odoo (only
the redemptions do, in pos_feedback_redemption). So the dashboard reads the
sheet directly and matches people on the last nine digits of their phone
number — the form takes the number however the customer types it.

Besides the main "Customer Feedback" form (Kenya), the sheet carries its own
forms for Uganda, Kampala and Sinza (the Sinza one in Swahili). They are read
too, so an attendant at those shops is credited for their customers' feedback.
"""
from __future__ import annotations


import pandas as pd
import streamlit as st

from lib import sheets_sync

SPREADSHEET_ID = "1gGcfdwsmjxsDz0PQlpHP_V3FEHncMpSbRzrsQ3gYdCY"
WORKSHEET = "Customer Feedback"

# Each tab and the header words its columns are found by: the first header
# containing any of the words, so a form question reworded around it still
# matches. The main tab's words are the ones the Request Hub is configured
# with in Odoo (denri_pos_request_hub.sheets_col_*).
_TABS = {
    WORKSHEET: {"Submitted": ["timestamp"], "Name": ["first name"],
                "Phone": ["phone number"], "Branch": ["which of our branches"],
                "Staff Rating": ["assistance you received from our staff"]},
    "Uganda":  {"Submitted": ["timestamp"], "Name": ["first name"],
                "Phone": ["phone"], "Branch": ["branch"],
                "Staff Rating": ["assistance you received in finding"]},
    "Kampala": {"Submitted": ["timestamp"], "Name": ["first name"],
                "Phone": ["phone"], "Branch": ["branch"],
                "Staff Rating": ["assistance you received in finding"]},
    "Sinza":   {"Submitted": ["timestamp"], "Name": ["jina la kwanza"],
                "Phone": ["nambari ya simu"], "Branch": ["tawi"],
                "Staff Rating": ["msaada uliopokea"]},
}
# Staff Rating: the 1-5 rating of the help the customer got from staff, the
# "Excellent customer service" measure on the attendant score cards (the
# Uganda, Kampala and Sinza forms ask about help finding/choosing the bag).
_FIELDS = ["Submitted", "Name", "Phone", "Branch", "Staff Rating"]


class FeedbackUnavailable(Exception):
    """The sheet could not be read; the message says what to do about it."""


def service_account_email() -> str:
    from lib import config
    info = config.get_gcp_service_account() or {}
    return info.get("client_email", "the dashboard's Google service account")


def phone_key(value) -> str | None:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits[-9:] if len(digits) >= 9 else None


def _parse_timestamps(raw: pd.Series) -> pd.Series:
    """The forms write day-first (02/10/2026 is 2 October); anything that
    does not fit that is tried as a general date."""
    parsed = pd.to_datetime(raw, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    rest = parsed.isna() & raw.str.strip().ne("")
    if rest.any():
        parsed[rest] = pd.to_datetime(raw[rest], errors="coerce", dayfirst=True)
    return parsed



def _read_tab(tabs: dict[str, list[list[str]]], title: str,
              words: dict[str, list[str]]) -> pd.DataFrame:
    found = {t.strip().lower(): t for t in tabs}
    real = found.get(title.lower())
    if real is None:
        return pd.DataFrame(columns=_FIELDS + ["Form"])
    values = tabs[real]
    if len(values) < 2:
        return pd.DataFrame(columns=_FIELDS + ["Form"])
    header = [h.strip().lower() for h in values[0]]
    rows = pd.DataFrame(values[1:])
    out = pd.DataFrame(index=rows.index)
    for field in _FIELDS:
        position = next((i for i, h in enumerate(header)
                         if any(w in h for w in words[field])), None)
        out[field] = rows[position].astype(str).str.strip() if position is not None else ""
    out["Form"] = real
    return out


def _not_shared(exc: Exception) -> bool:
    text = str(exc).upper()
    return isinstance(exc, PermissionError) or "403" in text or "PERMISSION" in text


@st.cache_data(ttl=600, show_spinner="Reading customer feedback…")
def load() -> pd.DataFrame:
    """Every feedback response: Submitted, Name, Phone, Branch, Staff Rating (1-5), Form, Phone Key."""
    try:
        # All four forms' tabs in one batched read.
        _, tabs = sheets_sync.read_tabs(SPREADSHEET_ID, list(_TABS))
        frames = [_read_tab(tabs, title, words) for title, words in _TABS.items()]
    except Exception as exc:
        if _not_shared(exc):
            raise FeedbackUnavailable(
                f"The feedback sheet is not shared with {service_account_email()}."
            ) from exc
        raise FeedbackUnavailable(f"Could not read the feedback sheet: {exc}") from exc

    out = pd.concat(frames, ignore_index=True)
    out = out[out["Submitted"].ne("")]
    out["Submitted"] = _parse_timestamps(out["Submitted"])
    out["Phone Key"] = out["Phone"].map(phone_key)
    out["Staff Rating"] = pd.to_numeric(out["Staff Rating"].str.extract(r"([1-5])")[0], errors="coerce")
    return out.reset_index(drop=True)
