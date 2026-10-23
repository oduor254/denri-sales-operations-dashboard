"""Shop manager (head) and shop attendant score cards, with weighted scores.

Every measure is scored out of 100 — the measure as a share of its target,
capped at 100 — and a card's score is the weighted average of its measures:

  Shop Manager (the shop's Head on its POS in Odoo)          weight
    Hit shop target         revenue / the shop's daily targets      30
    Basket value            revenue per order / KES 3,200           10
    Convert walk-ins        walk-in orders / walk-ins (footfall),
                            against the network's rate              10
    Convert WhatsApp leads  WhatsApp chats that bought / chats,
                            against the network's rate              10
    Response time           average first WhatsApp reply / 3 min     7
    Customer feedback       customers sending a form / 70%           8
    Training completion     the team's questionnaires answered       10
    Protect stock           stock-count variance: 0% scores 100,
                            10% or more off scores 0                 10

  Shop Attendant / BA
    Hit personal target     own sales / their daily target shares   30
    Bulk orders             bulk sales / 10% of their target         5
    Grow the basket         own revenue per order / KES 3,200       10
    Sale of new products    share of own sales from new bags,
                            against the network's share              4
    Conversion              own walk-in orders / customers served,
                            against the network's rate              15
    Respond fast            the shop's WhatsApp chats answered
                            within 3 min, on their WhatsApp days    10
    Customer service        average staff rating (1-5) from the
                            feedback of customers they served        7
    Customer feedback       their customers sending a form / 70%     4
    Capture the customer    their orders with a real customer phone  5
    Know the product        training score                           5
    Be there                on time on the days they worked          5

Roles change from day to day — at Hilton and KTDA one person prints at the
desk while the rest serve, Starmall sometimes does the same, and whoever is
on WhatsApp changes — so every measure is judged only on the days a person
actually did that work, read from the orders they were recorded on:
  * a day on the floor is a day they served a walk-in order there. Their
    target and their share of the shop's walk-ins (customers served) are
    the shop's for that day split equally between everyone on the floor;
  * a WhatsApp day is a day they served one of the shop's online orders.
    Respond Fast counts that shop's WhatsApp chats on those days only;
  * a desk day (ringing up, serving nobody) puts nothing on them.
A measure a person had no part in over the period (never on WhatsApp, no
clock-in data yet) is left out, and the card's score is the weighted
average of the measures they did — their weights scaled up to 100% — so
nobody is marked down for work they were not given.

Customers are known by phone number. The shops' own numbers — the ones on
the posters, typed in when a customer gives none — and any number shared by
five or more customer records are not customers: they count as details not
captured, and are left out of the feedback and service figures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NOT_RECORDED = "Not recorded"
NOT_SHOPS = {"Staff POS"}

# Each shop's own number, as printed on its posters (last nine digits).
SHOP_NUMBERS = {
    "118753780", "714156897", "759296069", "720060705", "712532047", "745390465",
    "118753776", "715598778", "115920422", "795176310", "798219855", "707392070",
    "707722454", "790833297", "703869724", "701012197", "118753771",
}
PLACEHOLDER_MIN_RECORDS = 5

BASKET_TARGET = 3200            # KES per order
FEEDBACK_TARGET = 70            # % of customers served sending a form
CORPORATE_SHARE = 0.10          # corporate sales expected as a share of the target
BULK_SHARE = 0.10               # bulk sales expected as a share of the target
BULK_MIN_BAGS = 5               # an order of this many bags or more is a bulk order
RESPONSE_TARGET_MIN = 3         # minutes to the first WhatsApp reply
STOCK_VARIANCE_ZERO_AT = 10.0   # stock-count variance % that scores 0
LATE_CREDIT = 0.5               # a late day counts half towards Be There

#   (score column, value column — already a % of target, weight)
HEAD_MEASURES = [
    ("Shop Target Score", "Target %", 30),
    ("Basket Value Score", "Basket Value %", 10),
    ("Walk-in Conversion Score", "Walk-in Conversion vs Network %", 10),
    ("WhatsApp Conversion Score", "WhatsApp Conversion vs Network %", 10),
    ("Response Score", "Response vs Target %", 7),
    ("Feedback Score", "Feedback of Target %", 8),
    ("Training Completion Score", "Training Completion %", 10),
    ("Stock Score", "Stock Accuracy %", 10),
]
ATTENDANT_MEASURES = [
    ("Personal Target Score", "Target %", 30),
    ("Bulk Orders Score", "Bulk %", 5),
    ("Basket Score", "Basket Value %", 10),
    ("New Products Score", "New Products vs Network %", 4),
    ("Conversion Score", "Conversion vs Network %", 15),
    ("Respond Fast Score", "Fast Reply %", 10),
    ("Service Score", "Service %", 7),
    ("Feedback Score", "Feedback of Target %", 4),
    ("Capture Score", "Capture %", 5),
    ("Training Score", "Training %", 5),
    ("Attendance Score", "Attendance %", 5),
]
HEAD_SCORE_COLUMNS = [m[0] for m in HEAD_MEASURES]
ATTENDANT_SCORE_COLUMNS = [m[0] for m in ATTENDANT_MEASURES]
WEIGHTS = {m[0]: m[2] for m in HEAD_MEASURES + ATTENDANT_MEASURES}
# Kept for anything still asking for the one list of score columns.
SCORE_COLUMNS = list(dict.fromkeys(HEAD_SCORE_COLUMNS + ATTENDANT_SCORE_COLUMNS))


def placeholders(*frames: pd.DataFrame) -> set[str]:
    """Phone numbers that are not a customer: the shops' own, and any shared
    by many customer records."""
    rows = pd.concat([f[["Customer", "Partner"]] for f in frames], ignore_index=True)
    shared = rows.groupby("Customer")["Partner"].nunique()
    return set(shared[shared >= PLACEHOLDER_MIN_RECORDS].index) | SHOP_NUMBERS


def prepare(orders: pd.DataFrame, not_customers: set[str]) -> pd.DataFrame:
    df = orders[~orders["Location"].isin(NOT_SHOPS)].copy()
    for col in ("Revenue", "Bags", "New Bag Revenue"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df.loc[df["Customer"].isin(not_customers), "Customer"] = None
    # A phone number on the order: 'partner:<id>' keys are records without one.
    df["Captured"] = df["Customer"].notna() & ~df["Customer"].astype(str).str.startswith("partner:")
    if "Order Type" not in df:
        df["Order Type"] = "walkin"
    df["Day"] = pd.to_datetime(df["Date"]).dt.date
    return df


def retention(current: pd.DataFrame, previous: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    """Unique customers in the previous period, per group, and how many of
    them bought again in this period at the same shop."""
    prev = previous.dropna(subset=["Customer"])[group + ["Customer"]].drop_duplicates()
    if prev.empty:
        return pd.DataFrame(columns=["Previous Customers", "Retained", "Retention %"])
    back = set(zip(current.dropna(subset=["Customer"])["Location"],
                   current.dropna(subset=["Customer"])["Customer"]))
    prev["Retained"] = [(loc, c) in back for loc, c in zip(prev["Location"], prev["Customer"])]
    out = prev.groupby(group).agg(**{"Previous Customers": ("Customer", "size"),
                                     "Retained": ("Retained", "sum")})
    out["Retention %"] = (out["Retained"] / out["Previous Customers"] * 100).round(1)
    return out


def gave_feedback(df: pd.DataFrame, responses: pd.DataFrame | None) -> pd.Series:
    """Per order: did its customer send feedback on or after the day they bought?"""
    if responses is None or responses.empty:
        return pd.Series(np.nan, index=df.index)
    first = (responses.dropna(subset=["Phone Key"])
             .groupby("Phone Key")["Submitted"].min().dt.normalize())
    fed_on = df["Customer"].map(first)
    return fed_on.notna() & (fed_on >= pd.to_datetime(df["Date"]))


def staff_ratings(df: pd.DataFrame, responses: pd.DataFrame | None) -> pd.Series:
    """Per order: the average 1-5 staff rating its customer gave on feedback
    forms sent on or after the day they bought (NaN when none)."""
    if responses is None or responses.empty or "Staff Rating" not in responses:
        return pd.Series(np.nan, index=df.index)
    rated = responses.dropna(subset=["Phone Key", "Staff Rating"])[
        ["Phone Key", "Submitted", "Staff Rating"]]
    orders = df[["Customer", "Date"]].reset_index().dropna(subset=["Customer"])
    hits = orders.merge(rated, left_on="Customer", right_on="Phone Key")
    hits = hits[hits["Submitted"].dt.normalize() >= pd.to_datetime(hits["Date"])]
    return hits.groupby("index")["Staff Rating"].mean().reindex(df.index)


def _customer_rates(df: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    """Repeat and feedback rates over the customers in each group."""
    known = df.dropna(subset=["Customer"])
    per = (known.groupby(group + ["Customer"])
           .agg(days=("Date", "nunique"), fed=("Feedback", "max"))
           .reset_index())
    per["repeat"] = per["days"] >= 2
    out = per.groupby(group).agg(Customers=("Customer", "size"), Repeat=("repeat", "sum"),
                                 Fed=("fed", "sum"), FedKnown=("fed", "count"))
    out["Repeat %"] = (out["Repeat"] / out["Customers"] * 100).round(1)
    out["Feedback %"] = np.where(out["FedKnown"] > 0, out["Fed"] / out["Customers"] * 100,
                                 np.nan).round(1)
    out["Feedback"] = np.where(out["FedKnown"] > 0, out["Fed"], np.nan)
    return out[["Customers", "Repeat %", "Feedback", "Feedback %"]]


def _day_key(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates).dt.date


def _pct(num, den) -> pd.Series:
    return (num / den.where(den > 0) * 100).round(1)


def weighted(table: pd.DataFrame, measures: list) -> pd.DataFrame:
    """Score each measure out of 100 and the card's weighted score.

    A measure with no value for a row (work they had no part in) is left out
    of that row's score: the weighted average runs over the measures they
    have, so their weights are scaled up to 100%. Weight Covered says how
    much of the full weighting the score stands on.
    """
    out = table.copy()
    total = pd.Series(0.0, index=out.index)
    covered = pd.Series(0.0, index=out.index)
    for column, value, weight in measures:
        values = out[value] if value in out else pd.Series(np.nan, index=out.index)
        out[column] = pd.to_numeric(values, errors="coerce").clip(lower=0, upper=100).round(0)
        has = out[column].notna()
        total += out[column].fillna(0) * weight
        covered += has * weight
    out["Score"] = (total / covered.where(covered > 0)).round(0)
    # Against the list's own total: the shop manager weights add up to 95.
    out["Weight Covered %"] = (covered / sum(m[2] for m in measures) * 100).round(0)
    return out


def _response_vs_target(minutes: pd.Series) -> pd.Series:
    out = pd.Series(np.where(minutes <= RESPONSE_TARGET_MIN, 100,
                             RESPONSE_TARGET_MIN / minutes * 100), index=minutes.index).round(1)
    return out.where(minutes.notna())


def _whatsapp_by_shop_day(wa: pd.DataFrame | None) -> pd.DataFrame:
    """Per shop and day: chats, chats answered within the 3 minutes, chats
    that bought."""
    cols = ["Location", "Day", "Chats", "Answered", "Fast", "Converted"]
    if wa is None or wa.empty:
        return pd.DataFrame(columns=cols)
    minutes = pd.to_numeric(wa["First Response (min)"], errors="coerce")
    frame = pd.DataFrame({
        "Location": wa["Branch"], "Day": _day_key(wa["Date"]),
        "Answered": minutes.notna(), "Fast": minutes <= RESPONSE_TARGET_MIN,
        "Converted": wa["Converted"].fillna(False).astype(bool),
    })
    return (frame.groupby(["Location", "Day"])
            .agg(Chats=("Answered", "size"), Answered=("Answered", "sum"),
                 Fast=("Fast", "sum"), Converted=("Converted", "sum"))
            .reset_index())


def _walkins_by_shop_day(walkins: pd.DataFrame | None) -> pd.Series:
    if walkins is None or walkins.empty:
        return pd.Series(dtype=float)
    w = walkins.assign(Day=_day_key(walkins["Date"]))
    return w.groupby(["Location", "Day"])["Walk-ins"].sum()


def _network_conversion(df: pd.DataFrame, walk: pd.Series) -> float | None:
    """Walk-in orders over walk-ins, network-wide, on shop-days with footfall."""
    if walk.empty:
        return None
    orders = (df[df["Order Type"] == "walkin"].groupby(["Location", "Day"]).size()
              .reindex(walk.index).fillna(0))
    counted = walk > 0
    return float(orders[counted].sum() / walk[counted].sum() * 100) if counted.any() else None


def shop_cards(df: pd.DataFrame, previous: pd.DataFrame, daily_targets: pd.DataFrame,
               training: pd.Series | None = None,
               response: pd.DataFrame | None = None, *,
               training_completion: pd.Series | None = None,
               walkins: pd.DataFrame | None = None,
               wa: pd.DataFrame | None = None,
               stock: pd.DataFrame | None = None,
               heads: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per shop, scored as its shop manager's (Head's) card.

    daily_targets is targets.daily(): Location, Date, Target for each day so
    far. walkins is footfall per Location/Date; wa the WhatsApp chats
    (queries.WA_INTERACTIONS); stock queries.STOCK_COUNTS; heads
    queries.SHOP_HEADS."""
    g = df.groupby("Location")
    staff = df[df["Served By"] != NOT_RECORDED]
    table = pd.DataFrame({
        "Revenue": g["Revenue"].sum(),
        "Corporate Revenue": g.apply(lambda f: f.loc[f["Corporate"], "Revenue"].sum(),
                                     include_groups=False),
        "Orders": g.size(),
        "Bags": g["Bags"].sum(),
        "Attendants": staff.groupby("Location")["Served By"].nunique(),
    })
    table["Attendants"] = table["Attendants"].fillna(0).astype(int)
    table["Target"] = daily_targets.groupby("Location")["Target"].sum().reindex(table.index)
    table["Target %"] = _pct(table["Revenue"], table["Target"])
    table["Basket Value"] = (table["Revenue"] / table["Orders"]).round(0)
    table["Basket Value %"] = (table["Basket Value"] / BASKET_TARGET * 100).round(1)
    table["Basket Size"] = (table["Bags"] / table["Orders"]).round(2)
    table["Revenue per Attendant"] = (table["Revenue"]
                                      / table["Attendants"].where(table["Attendants"] > 0)).round(0)
    table = table.join(_customer_rates(df, ["Location"]))
    table = table.join(retention(df, previous, ["Location"]))
    table["Feedback of Target %"] = (table["Feedback %"] / FEEDBACK_TARGET * 100).round(1)
    table["Training %"] = training.reindex(table.index).round(1) if training is not None else np.nan
    table["Training Completion %"] = (training_completion.reindex(table.index).round(1)
                                      if training_completion is not None else np.nan)

    # Walk-in conversion: walk-in orders over footfall, on days footfall is in.
    walk = _walkins_by_shop_day(walkins)
    network = _network_conversion(df, walk)
    if not walk.empty:
        orders = (df[df["Order Type"] == "walkin"].groupby(["Location", "Day"]).size()
                  .reindex(walk.index).fillna(0))
        counted = walk > 0
        table["Walk-ins"] = walk[counted].groupby(level="Location").sum().reindex(table.index)
        table["Walk-in Orders"] = orders[counted].groupby(level="Location").sum().reindex(table.index)
        table["Walk-in Conversion %"] = _pct(table["Walk-in Orders"], table["Walk-ins"])
    else:
        table["Walk-ins"] = table["Walk-in Orders"] = table["Walk-in Conversion %"] = np.nan
    table["Walk-in Conversion vs Network %"] = (
        (table["Walk-in Conversion %"] / network * 100).round(1) if network else np.nan)

    # WhatsApp: conversion against the network, and response time.
    chats = _whatsapp_by_shop_day(wa).groupby("Location")[["Chats", "Converted"]].sum()
    table["WhatsApp Chats"] = chats["Chats"].reindex(table.index)
    table["WhatsApp Converted"] = chats["Converted"].reindex(table.index)
    table["WhatsApp Conversion %"] = _pct(table["WhatsApp Converted"], table["WhatsApp Chats"])
    wa_network = (chats["Converted"].sum() / chats["Chats"].sum() * 100) if chats["Chats"].sum() else None
    table["WhatsApp Conversion vs Network %"] = (
        (table["WhatsApp Conversion %"] / wa_network * 100).round(1) if wa_network else np.nan)
    if response is not None and not response.empty:
        table = table.join(response)
        table["Response vs Target %"] = _response_vs_target(table["Response Time (min)"])
    else:
        table["Response Time (min)"] = table["Chats"] = table["Response vs Target %"] = np.nan

    # Stock: variance found by the latest counts.
    if stock is not None and not stock.empty:
        s = stock.set_index("Location")
        variance = (s["Variance Qty"] / s["System Qty"].where(s["System Qty"] > 0) * 100)
        table["Stock Variance %"] = variance.reindex(table.index).round(2)
        table["Last Stock Count"] = s["Last Count"].reindex(table.index)
        table["Stock Accuracy %"] = (100 - table["Stock Variance %"] * 100 / STOCK_VARIANCE_ZERO_AT
                                     ).clip(lower=0).round(1)
    else:
        table["Stock Variance %"] = table["Last Stock Count"] = table["Stock Accuracy %"] = np.nan

    table["Head"] = (heads.set_index("Location")["Head"].reindex(table.index)
                     if heads is not None and not heads.empty else None)
    table = weighted(table, HEAD_MEASURES)
    table = table.reset_index().rename(columns={"index": "Location"})
    return table.sort_values("Score", ascending=False)


def attendant_cards(df: pd.DataFrame, previous: pd.DataFrame, daily_targets: pd.DataFrame,
                    training: pd.Series | None = None, *,
                    walkins: pd.DataFrame | None = None,
                    wa: pd.DataFrame | None = None,
                    attendance: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per attendant or BA (per shop they served at).

    Day by day: the shop's daily target is shared equally between everyone
    on the floor that day, and a person's target is the sum of their shares
    over the days they worked. Revenue is their own sales, so Target % is
    what they generated over the target they were given. The other measures
    follow the daily roles described in the module docstring.
    """
    staffed = df[df["Served By"] != NOT_RECORDED].copy()
    keys = ["Served By", "Location"]
    team = staffed.groupby(["Location", "Day"])["Served By"].nunique().rename("Team")
    day_target = (daily_targets.assign(Day=_day_key(daily_targets["Date"]))
                  .groupby(["Location", "Day"])["Target"].sum().rename("Day Target"))
    worked = (staffed[["Location", "Day", "Served By"]].drop_duplicates()
              .join(team, on=["Location", "Day"]).join(day_target, on=["Location", "Day"]))
    worked["Target Share"] = worked["Day Target"] / worked["Team"]
    g = staffed.groupby(keys)
    by_person = worked.groupby(keys)
    bulk = staffed["Corporate"] | (staffed["Bags"] >= BULK_MIN_BAGS)
    table = pd.DataFrame({
        "Revenue": g["Revenue"].sum().round(0),
        "Days Worked": by_person["Day"].nunique(),
        "Avg Team": by_person["Team"].mean().round(1),
        "Target": by_person["Target Share"].sum(min_count=1).round(0),
        "Orders": g.size(),
        "Bags": g["Bags"].sum(),
        "Corporate Revenue": staffed[staffed["Corporate"]].groupby(keys)["Revenue"].sum(),
        "Bulk Revenue": staffed[bulk].groupby(keys)["Revenue"].sum(),
        "New Bag Revenue": g["New Bag Revenue"].sum() if "New Bag Revenue" in staffed else np.nan,
        "Captured Orders": g["Captured"].sum(),
    })
    for col in ("Corporate Revenue", "Bulk Revenue"):
        table[col] = table[col].fillna(0)
    table["Target %"] = _pct(table["Revenue"], table["Target"])
    table["Revenue per Day"] = (table["Revenue"] / table["Days Worked"]).round(0)
    table["Basket Value"] = (table["Revenue"] / table["Orders"]).round(0)
    table["Basket Value %"] = (table["Basket Value"] / BASKET_TARGET * 100).round(1)
    table["Basket Size"] = (table["Bags"] / table["Orders"]).round(2)
    table["Corporate Target"] = (table["Target"] * CORPORATE_SHARE).round(0)
    table["Corporate %"] = _pct(table["Corporate Revenue"], table["Corporate Target"])
    table["Bulk Target"] = (table["Target"] * BULK_SHARE).round(0)
    table["Bulk %"] = _pct(table["Bulk Revenue"], table["Bulk Target"])

    # New products: share of own sales from new bags, against the network's.
    table["New Products %"] = _pct(table["New Bag Revenue"], table["Revenue"])
    net_new = staffed["New Bag Revenue"].sum() / staffed["Revenue"].sum() * 100 \
        if "New Bag Revenue" in staffed and staffed["Revenue"].sum() else None
    table["New Products vs Network %"] = ((table["New Products %"] / net_new * 100).round(1)
                                          if net_new else np.nan)

    # Conversion: walk-in orders over customers served — the shop's walk-ins
    # that day split between everyone serving walk-ins that day.
    walk = _walkins_by_shop_day(walkins)
    network = _network_conversion(df, walk)
    floor = staffed[staffed["Order Type"] == "walkin"]
    if not walk.empty and not floor.empty:
        floor_team = floor.groupby(["Location", "Day"])["Served By"].nunique()
        served = floor.groupby(keys + ["Day"]).size().rename("Walk-in Orders").reset_index()
        served["Shop Walk-ins"] = [walk.get((l, d), np.nan) for l, d in zip(served["Location"], served["Day"])]
        served["Team"] = [floor_team.get((l, d)) for l, d in zip(served["Location"], served["Day"])]
        served = served[served["Shop Walk-ins"] > 0]
        served["Customers Served"] = served["Shop Walk-ins"] / served["Team"]
        conv = served.groupby(keys)[["Walk-in Orders", "Customers Served"]].sum()
        table["Walk-in Orders"] = conv["Walk-in Orders"].reindex(table.index)
        table["Customers Served"] = conv["Customers Served"].reindex(table.index).round(1)
        table["Conversion %"] = _pct(table["Walk-in Orders"], table["Customers Served"])
    else:
        table["Walk-in Orders"] = table["Customers Served"] = table["Conversion %"] = np.nan
    table["Conversion vs Network %"] = ((table["Conversion %"] / network * 100).round(1)
                                        if network else np.nan)

    # Respond fast: the shop's chats answered within 3 minutes, on the days
    # the person was on WhatsApp (served the shop's online orders).
    chats = _whatsapp_by_shop_day(wa).set_index(["Location", "Day"])
    online = staffed[staffed["Order Type"] == "online"][keys + ["Day"]].drop_duplicates()
    if not chats.empty and not online.empty:
        on = online.join(chats[["Answered", "Fast"]], on=["Location", "Day"]).dropna(subset=["Answered"])
        fast = on.groupby(keys)[["Answered", "Fast"]].sum()
        table["WhatsApp Days"] = online.groupby(keys)["Day"].nunique().reindex(table.index)
        table["Chats Answered"] = fast["Answered"].reindex(table.index)
        table["Fast Reply %"] = _pct(fast["Fast"].reindex(table.index), table["Chats Answered"])
    else:
        table["WhatsApp Days"] = table["Chats Answered"] = table["Fast Reply %"] = np.nan

    # Customer service: the staff rating their customers gave (1-5).
    if "Staff Rating" in staffed:
        rated = staffed.dropna(subset=["Staff Rating", "Customer"]).drop_duplicates(keys + ["Customer"])
        table["Service Rating"] = rated.groupby(keys)["Staff Rating"].mean().reindex(table.index).round(2)
        table["Ratings"] = rated.groupby(keys).size().reindex(table.index)
    else:
        table["Service Rating"] = table["Ratings"] = np.nan
    table["Service %"] = (table["Service Rating"] / 5 * 100).round(1)

    table = table.join(_customer_rates(staffed, keys).drop(columns=["Repeat %"]))
    table["Feedback of Target %"] = (table["Feedback %"] / FEEDBACK_TARGET * 100).round(1)
    table["Capture %"] = _pct(table["Captured Orders"], table["Orders"])
    table["Training %"] = training.reindex(table.index).round(1) if training is not None else np.nan

    # Be there: on time on the days they worked, for people who clock in.
    table["Attendance %"] = np.nan
    table["Late Days"] = np.nan
    if attendance is not None and not attendance.empty and "User ID" in staffed:
        att = attendance.assign(Day=_day_key(attendance["Date"]))
        clocked = set(att.loc[att["Status"].isin(["present", "late"]), "User ID"])
        status = att.set_index(["User ID", "Day"])["Status"]
        days = (staffed.dropna(subset=["User ID"])[keys + ["Day", "User ID"]]
                .drop_duplicates(keys + ["Day"]))
        days = days[days["User ID"].isin(clocked)]
        if not days.empty:
            days["Status"] = [status.get((u, d)) for u, d in zip(days["User ID"], days["Day"])]
            days["Credit"] = days["Status"].map({"present": 1.0, "late": LATE_CREDIT,
                                                 "wfh": 1.0, "ooo": 1.0}).fillna(0.0)
            days["Late"] = days["Status"].eq("late")
            per = days.groupby(keys).agg(credit=("Credit", "mean"), late=("Late", "sum"))
            table["Attendance %"] = (per["credit"] * 100).reindex(table.index).round(1)
            table["Late Days"] = per["late"].reindex(table.index)

    table = weighted(table, ATTENDANT_MEASURES)
    table = table.reset_index().rename(columns={"Served By": "Attendant / BA"})
    return table.sort_values("Score", ascending=False)
