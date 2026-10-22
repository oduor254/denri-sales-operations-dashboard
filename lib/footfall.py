"""Footfall, walk-in orders and revenue: live from Odoo, balanced against the
shared "Reporting" Google Sheet.

Where each figure comes from, per shop and day (combine() applies this):
  * Bought (purchased walk-ins): the POS's walk-in orders, live — the same
    count as the Shop Ranking's "Walk-in Orders". The sheet's purchased column
    is copied from it (every day of Oct 2026 matches), so the sheet is used
    only for a day the system has nothing for.
  * Walk-ins (total footfall): the footfall module's count
    (denri_footfall_count), keyed at the till at the end of the day. Until
    every till does that reliably the sheet fills the gaps: a day with no
    count, or a count below the day's walk-in orders (a count keyed mid-day
    and never updated — Starmall 15 Oct 2026: 22 counted, 24 bought, 30 on
    the sheet).
  * Revenue: the sheet once finance has reconciled the day (the sheet's day
    total is filled in), the POS takings until then. The page shows both and
    the difference, so a reconciliation can be checked against the till.

The sheet's layouts:

Shop teams log every walk-in on the "Footfall" tab: one row per shop, and per
day a four-column block (WALKINS PURCHASED, W.NOT PURCHASED, TOTAL, C. RATE)
under a dd/mm/yyyy date in row 1. Every few days there is also a "WEEK n"
block of weekly totals — skipped here, since the weeks are re-added from the
days. The date label sits over the block's second column most of the time
and over its first now and then, so a block is found by its header and its
date by looking across all four of its columns.

The "Revenue" tab is one row per day (m/d/yyyy) with a column per shop, the
same figures the weekly "Sales vs Conversion Rate" workbook is built from.
Its NAIROBI column is Starmall and TANZANIA is Sinza; WEBSITE, CORPORATE and
JUMIA sell without walk-ins and are kept apart as "Other Channels".

Shop targets are not in the sheet: the workbook's "Daily Target" row is the
Odoo daily target (queries.SALES_TARGET_ROWS), so that is what is used.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

from lib import sheets_sync
from lib.feedback import _not_shared, service_account_email

SPREADSHEET_ID = "1S4kRCl1KWMTWKaSF8NSIXjDiJmvREzv7hm4vDnowdqs"
FOOTFALL_TAB = "Footfall"
REVENUE_TAB = "Revenue"

# The shops in the order the business lists them (the Footfall tab's order).
SHOP_ORDER = ["Starmall", "Mombasa", "Nakuru", "Eldoret", "Kisumu", "Meru", "Thika",
              "Hazina", "Kitengela", "Nanyuki", "Kakamega", "Hilton", "Sinza", "Uganda",
              "Kisii", "KTDA", "Busia", "Rongai"]

# Revenue-tab column -> shop name used on the page.
_REVENUE_NAMES = {"NAIROBI": "Starmall", "TANZANIA": "Sinza", "KTDA": "KTDA"}
OTHER_CHANNELS = ["Website", "Corporate", "Jumia"]
NO_TILL = {"Corporate"}

# Odoo target / POS location -> shop name used on the page.
TARGET_NAMES = {"Ktda": "KTDA"}
SYSTEM_NAMES = {"Ktda": "KTDA"}

RECONCILED, LIVE = "Reconciled (sheet)", "Live (POS)"
COUNTED, SHEET_COUNT, LOW_COUNT = "Footfall module", "Sheet", "Footfall module (below orders)"
NOT_COUNTED = "Not counted"

_DATE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")


class SheetUnavailable(Exception):
    """The sheet could not be read; the message says what to do about it."""


def _number(value) -> float | None:
    text = str(value or "").replace(",", "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _shop_name(raw: str) -> str:
    name = raw.strip()
    upper = name.upper()
    if upper in _REVENUE_NAMES:
        return _REVENUE_NAMES[upper]
    return name if upper != name else name.title()


def _parse_footfall(values: list[list[str]]) -> pd.DataFrame:
    """Wide day blocks -> one row per Shop and Date: Purchased, Not Purchased."""
    if len(values) < 3:
        return pd.DataFrame(columns=["Shop", "Date", "Purchased", "Not Purchased"])
    labels, header = values[0], values[1]
    width = max(len(r) for r in values)
    labels = labels + [""] * (width - len(labels))
    starts = [i for i, h in enumerate(header) if h.strip().upper().startswith("WALKINS PURCHASED")]

    rows = []
    for s in starts:
        block = [labels[i].strip() for i in range(s, min(s + 4, width))]
        if any("WEEK" in b.upper() for b in block):
            continue
        day = next((b for b in block if _DATE.match(b)), None)
        if day is None:
            continue
        day = datetime.strptime(day, "%d/%m/%Y").date()
        for r in values[2:]:
            shop = r[0].strip() if r else ""
            if not shop or shop.lower() == "total":
                continue
            bought = _number(r[s]) if s < len(r) else None
            missed = _number(r[s + 1]) if s + 1 < len(r) else None
            if bought is None and missed is None:
                continue
            rows.append({"Shop": shop, "Date": day,
                         "Purchased": bought or 0.0, "Not Purchased": missed or 0.0})
    out = pd.DataFrame(rows, columns=["Shop", "Date", "Purchased", "Not Purchased"])
    # A day logged twice keeps its last entry, as the sheet's own totals would.
    return out.drop_duplicates(["Shop", "Date"], keep="last")


def _parse_revenue(values: list[list[str]]) -> pd.DataFrame:
    """Daily rows x shop columns -> one row per Shop and Date: Revenue."""
    header_at = next((i for i, r in enumerate(values)
                      if any(c.strip().upper() == "DATE" for c in r)), None)
    if header_at is None:
        return pd.DataFrame(columns=["Shop", "Date", "Revenue"])
    header = values[header_at]
    date_col = next(i for i, c in enumerate(header) if c.strip().upper() == "DATE")
    shops = {i: _shop_name(c) for i, c in enumerate(header) if c.strip() and i != date_col}

    rows = []
    for r in values[header_at + 1:]:
        raw = r[date_col].strip() if date_col < len(r) else ""
        if not _DATE.match(raw):
            continue  # the TOTAL row, blanks
        day = datetime.strptime(raw, "%m/%d/%Y").date()
        for i, shop in shops.items():
            amount = _number(r[i]) if i < len(r) else None
            if amount is not None:
                rows.append({"Shop": shop, "Date": day, "Revenue": amount})
    return pd.DataFrame(rows, columns=["Shop", "Date", "Revenue"])


@st.cache_data(ttl=600, show_spinner="Reading footfall and revenue…")
def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(footfall, revenue): footfall per Shop/Date, revenue per Shop/Date
    (shops and Other Channels alike)."""
    try:
        # Both tabs in one batched read.
        _, tabs = sheets_sync.read_tabs(SPREADSHEET_ID, [FOOTFALL_TAB, REVENUE_TAB])
    except Exception as exc:  # not shared, quota, Google being unavailable
        if _not_shared(exc):
            raise SheetUnavailable(
                f"The Reporting sheet is not shared with {service_account_email()}."
            ) from exc
        raise SheetUnavailable(f"Could not read the Reporting sheet: {exc}") from exc
    found = {t.strip().lower(): v for t, v in tabs.items()}
    missing = [t for t in (FOOTFALL_TAB, REVENUE_TAB) if t.lower() not in found]
    if missing:
        raise SheetUnavailable(f"The Reporting sheet has no {' or '.join(missing)} tab.")
    return _parse_footfall(found[FOOTFALL_TAB.lower()]), _parse_revenue(found[REVENUE_TAB.lower()])


def daily_targets(rows: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Odoo daily targets laid out per Shop and Date over [start, end]."""
    cols = ["Shop", "Date", "Target"]
    if rows is None or rows.empty:
        return pd.DataFrame(columns=cols)
    day_rows = rows[rows["Kind"] == "day"].copy()
    day_rows["Starts"] = pd.to_datetime(day_rows["Starts"]).dt.date
    day_rows["Ends"] = pd.to_datetime(day_rows["Ends"]).dt.date
    out = []
    d = start
    while d <= end:
        hit = day_rows[(day_rows["Starts"] <= d) & (day_rows["Ends"] >= d)]
        for loc, amount in hit.groupby("Location")["Target"].sum().items():
            out.append({"Shop": TARGET_NAMES.get(loc, loc), "Date": d, "Target": amount})
        d += timedelta(days=1)
    return pd.DataFrame(out, columns=cols)


_SYSTEM_COLS = ["Shop", "Date", "System Revenue", "Walk-in Orders", "Footfall"]


def prepare_system(rows: pd.DataFrame | None) -> pd.DataFrame:
    """queries.FOOTFALL_SYSTEM_DAILY with the page's shop names and plain dates."""
    if rows is None or rows.empty:
        return pd.DataFrame(columns=_SYSTEM_COLS)
    out = rows.copy()
    out["Shop"] = out["Shop"].replace(SYSTEM_NAMES)
    out["Date"] = pd.to_datetime(out["Date"]).dt.date
    for col in ("System Revenue", "Walk-in Orders", "Footfall"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    return out[_SYSTEM_COLS]


def reconciled_days(rev: pd.DataFrame) -> set[date]:
    """Days finance has filled in on the Revenue tab (the day's total is not
    zero). The tab carries a zero row for every day of the year in advance."""
    totals = rev.groupby("Date")["Revenue"].sum()
    return set(totals[totals > 0].index)


def _in_range(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    return df[(df["Date"] >= start) & (df["Date"] <= end)]


def combine(foot: pd.DataFrame, rev: pd.DataFrame, system: pd.DataFrame,
            targets: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """One row per shop and day in the range, with the sources picked as the
    module docstring sets out: Revenue (and Revenue Source), Target, Walk-ins
    (and Footfall Source), Purchased, Not Purchased, plus the raw System
    Revenue / Sheet Revenue for the reconciliation check.

    Conversion and Sales per Walk-in only count shop-days whose footfall is
    in (Counted Purchased, Logged Revenue): today's walk-in orders arrive all
    day, the footfall count only at closing.

    Only days with something recorded anywhere are kept, so the days ahead
    don't count their targets before anything has been sold; on those days
    every shop counts its target, sales or not."""
    shops = set(SHOP_ORDER) | set(foot["Shop"])
    keys = ["Shop", "Date"]
    sheet_foot = _in_range(foot, start, end).rename(
        columns={"Purchased": "Sheet Bought", "Not Purchased": "Sheet Missed"})
    sheet_rev = _in_range(rev[rev["Shop"].isin(shops)], start, end).rename(
        columns={"Revenue": "Sheet Revenue"})
    sys_rows = _in_range(system[system["Shop"].isin(shops)], start, end)
    out = (sheet_foot.set_index(keys)
           .join(sheet_rev.set_index(keys), how="outer")
           .join(sys_rows.set_index(keys), how="outer")
           .join(_in_range(targets[targets["Shop"].isin(shops)], start, end).set_index(keys),
                 how="outer")
           .reset_index())

    done = reconciled_days(rev)
    on_system = set(system.loc[(system["System Revenue"] != 0) | (system["Walk-in Orders"] > 0), "Date"])
    reconciled = out["Date"].isin(done)
    out["Revenue"] = out["Sheet Revenue"].fillna(0.0).where(reconciled, out["System Revenue"].fillna(0.0))
    out["Revenue Source"] = reconciled.map({True: RECONCILED, False: LIVE})

    sheet_walk = out["Sheet Bought"] + out["Sheet Missed"].fillna(0.0)
    out["Purchased"] = out["Walk-in Orders"].fillna(0.0).where(
        out["Date"].isin(on_system), out["Sheet Bought"].fillna(0.0))
    counted = out["Footfall"]
    use_count = counted.notna() & (counted >= out["Purchased"])
    use_sheet = ~use_count & sheet_walk.notna() & (sheet_walk >= out["Purchased"]) & (sheet_walk > 0)
    low_count = ~use_count & ~use_sheet & counted.notna()
    out["Walk-ins"] = 0.0
    out.loc[use_count, "Walk-ins"] = counted[use_count]
    out.loc[use_sheet, "Walk-ins"] = sheet_walk[use_sheet]
    # A count below the orders and nothing better: everyone who was counted
    # bought, at least — floored so the day can't read over 100%.
    out.loc[low_count, "Walk-ins"] = out.loc[low_count, "Purchased"]
    out["Footfall Source"] = None
    out.loc[use_count, "Footfall Source"] = COUNTED
    out.loc[use_sheet, "Footfall Source"] = SHEET_COUNT
    out.loc[low_count, "Footfall Source"] = LOW_COUNT
    out.loc[out["Footfall Source"].isna() & (out["Purchased"] > 0), "Footfall Source"] = NOT_COUNTED

    logged = out["Walk-ins"] > 0
    out["Counted Purchased"] = out["Purchased"].where(logged, 0.0)
    out["Not Purchased"] = (out["Walk-ins"] - out["Counted Purchased"]).clip(lower=0)
    out["Logged Revenue"] = out["Revenue"].where(logged, 0.0)

    active = set(out.loc[(out["Walk-ins"] > 0) | (out["Revenue"] != 0) | (out["Purchased"] > 0), "Date"])
    return out[out["Date"].isin(active)].reset_index(drop=True)


def footfall_pending(df: pd.DataFrame) -> list[date]:
    """Days in df with walk-in orders or sales but no footfall in at any shop yet."""
    by_day = df.groupby("Date")[["Walk-ins", "Purchased", "Revenue"]].sum()
    return sorted(by_day[(by_day["Walk-ins"] == 0)
                         & ((by_day["Purchased"] > 0) | (by_day["Revenue"] > 0))].index)


def other_channels(rev: pd.DataFrame, system: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Website, Corporate and Jumia revenue per day over the range: the sheet's
    once reconciled, the POS's until then (Corporate has no till, so it shows
    only once reconciled)."""
    done = reconciled_days(rev)
    sheet = _in_range(rev[rev["Shop"].isin(OTHER_CHANNELS)], start, end)
    sheet = sheet[sheet["Date"].isin(done)]
    live = _in_range(system[system["Shop"].isin(OTHER_CHANNELS)], start, end)
    live = live[~live["Date"].isin(done)].rename(columns={"System Revenue": "Revenue"})
    out = pd.concat([sheet, live[["Shop", "Date", "Revenue"]]], ignore_index=True)
    return out[out["Revenue"] != 0]


def reconciliation(rev: pd.DataFrame, system: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Per shop/channel and reconciled day: POS takings against the sheet's
    reconciled revenue. Variance is sheet minus POS — what reconciliation
    added (+) or took off (−) the till's figure. Corporate is left out: it
    has no till, so all of it would read as variance."""
    done = reconciled_days(rev)
    keys = ["Shop", "Date"]
    sheet = _in_range(rev[~rev["Shop"].isin(NO_TILL)], start, end)
    sheet = sheet[sheet["Date"].isin(done)].rename(columns={"Revenue": "Sheet Revenue"})
    live = _in_range(system, start, end)
    live = live[live["Date"].isin(done) & (live["Shop"] != "Staff POS")]
    out = (sheet.set_index(keys).join(live.set_index(keys)[["System Revenue"]], how="outer")
           .reset_index())
    out["Sheet Revenue"] = out["Sheet Revenue"].fillna(0.0)
    out["System Revenue"] = out["System Revenue"].fillna(0.0)
    out["Variance"] = out["Sheet Revenue"] - out["System Revenue"]
    return out[(out["Sheet Revenue"] != 0) | (out["System Revenue"] != 0)]


def summarise(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Sum the counts per group and work out the ratios on the sums, so a
    week's conversion is its walk-ins weighted, not an average of days."""
    counts = ["Revenue", "Logged Revenue", "Walk-ins", "Purchased", "Counted Purchased",
              "Not Purchased"]
    agg = df.groupby(by, as_index=False)[counts + ["Target"]].sum(min_count=1)
    for col in counts:
        agg[col] = agg[col].fillna(0.0)
    walk = agg["Walk-ins"].where(agg["Walk-ins"] > 0)
    bought = agg["Counted Purchased"].where(agg["Counted Purchased"] > 0)
    agg["Conversion %"] = agg["Counted Purchased"] / walk * 100
    agg["% of Target"] = agg["Revenue"] / agg["Target"].where(agg["Target"] > 0) * 100
    agg["Sales per Walk-in"] = agg["Logged Revenue"] / walk
    agg["Sales per Buyer"] = agg["Logged Revenue"] / bought
    return agg


def week_start(d: date) -> date:
    """The Sunday that starts d's week (Sunday–Saturday, as the workbook)."""
    return d - timedelta(days=(d.weekday() + 1) % 7)


def week_number(d: date) -> int:
    """The workbook's week number: week 1 holds 1 January, weeks run
    Sunday–Saturday — so 4–10 Oct 2026 is Week 41."""
    first = week_start(date(d.year, 1, 1))
    return (week_start(d) - first).days // 7 + 1


def shop_order(found) -> list[str]:
    found = set(found)
    return [s for s in SHOP_ORDER if s in found] + sorted(found - set(SHOP_ORDER))
