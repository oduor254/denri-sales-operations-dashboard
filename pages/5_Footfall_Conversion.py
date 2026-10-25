"""Footfall Conversion — walk-ins, how many bought, and the sales they brought,
against each shop's daily target. The online version of the weekly "Sales vs
Conversion Rate" workbook: walk-in orders, takings and footfall live from
Odoo, balanced against the Reporting Google Sheet (see lib/footfall.py for
which source wins where), targets from Odoo."""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from lib import db, filters, grid, queries, theme
from lib import footfall as ff

st.set_page_config(page_title="Footfall Conversion · Denri Sales Operations Dashboard",
                   page_icon="🚶", layout="wide")

st.title("🚶 Footfall Conversion")
st.caption("Walk-ins, how many bought, and the sales they brought against each shop's daily "
           "target — live from the POS and footfall module, with revenue switching to the "
           "finance-reconciled figure from the Reporting sheet once a day is reconciled")

TODAY = date.today()
ALL = "All Shops"
WEEKS_SHOWN = 8

col_period, col_loc, col_refresh = st.columns([2, 2, 1])
with col_period:
    start, end = filters.date_range_control("footfall", default="Last 7 Days")
with col_refresh:
    if db.refresh_button(key="footfall_refresh"):
        ff.load.clear()

days = (end - start).days + 1
prev_end = start - timedelta(days=1)
prev_start = prev_end - timedelta(days=days - 1)
weeks_from = ff.week_start(TODAY) - timedelta(weeks=WEEKS_SHOWN - 1)
history_from = min(prev_start, weeks_from)
history_to = max(end, TODAY)

# Each source is optional: without the sheet the page runs live on Odoo alone
# (no reconciled revenue, no footfall fallback); without Postgres it runs on
# the sheet alone, as the workbook did.
try:
    foot, rev = ff.load()
    sheet_problem = None
except ff.SheetUnavailable as exc:
    foot = pd.DataFrame(columns=["Shop", "Date", "Purchased", "Not Purchased"])
    rev = pd.DataFrame(columns=["Shop", "Date", "Revenue"])
    sheet_problem = (f"{exc} Showing live POS figures only — share the Reporting sheet with "
                     f"**{ff.service_account_email()}** as a Viewer for reconciled revenue.")

try:
    params = {"start_date": history_from, "end_date": history_to}
    system = ff.prepare_system(db.run_query(queries.FOOTFALL_SYSTEM_DAILY, params))
    targets = ff.daily_targets(db.run_query(queries.SALES_TARGET_ROWS, params),
                               history_from, history_to)
    system_problem = None
except Exception as exc:
    system = ff.prepare_system(None)
    targets = pd.DataFrame(columns=["Shop", "Date", "Target"])
    system_problem = f"Could not reach Odoo ({exc}). Showing the Reporting sheet only, without targets."

if sheet_problem and system_problem:
    st.error(f"{sheet_problem}\n\n{system_problem}", icon="🚫")
    st.stop()
for problem in (sheet_problem, system_problem):
    if problem:
        st.warning(problem, icon="⚠️")

current = ff.combine(foot, rev, system, targets, start, end)
previous = ff.combine(foot, rev, system, targets, prev_start, prev_end)
recent = ff.combine(foot, rev, system, targets, weeks_from, TODAY)

shops = ff.shop_order(pd.concat([current["Shop"], recent["Shop"]]).unique())
with col_loc:
    location = st.selectbox("Shop", [ALL] + shops, key="footfall_location")


def _mine(df: pd.DataFrame) -> pd.DataFrame:
    return df if location == ALL else df[df["Shop"] == location]


def _total(df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None
    df = df.assign(_all=1)
    return ff.summarise(df, ["_all"]).iloc[0]


def _kes(v) -> str:
    return "—" if v is None or pd.isna(v) else f"KES {v:,.0f}"


def _pct(v) -> str:
    return "—" if v is None or pd.isna(v) else f"{v:.1f}%"


def _change(now, before) -> float | None:
    if before is None or pd.isna(before) or before == 0 or now is None or pd.isna(now):
        return None
    return (now - before) / abs(before) * 100


period_label = (f"{start:%a %d %b}" if days == 1 else f"{start:%d %b} – {end:%d %b %Y}")
st.subheader(f"{location} — {period_label}")

now, before = _total(_mine(current)), _total(_mine(previous))
if now is None:
    st.info("No sales, walk-in orders or footfall recorded for this period yet.")
    st.stop()

pending = ff.footfall_pending(_mine(current))
if TODAY in pending:
    st.info("Today's footfall is counted at closing, so today's sales and walk-in orders show "
            "live but don't count towards conversion or sales per walk-in until then.", icon="⏳")
past = [d for d in pending if d != TODAY]
if past:
    st.warning("No footfall recorded at all for " + ", ".join(f"{d:%a %d %b}" for d in past)
               + ". Those days count towards sales and walk-in orders, but not conversion.",
               icon="⚠️")

prev_name = "previous day" if days == 1 else f"previous {days} days"


def _matching(logged) -> pd.Series | None:
    """The previous period cut to the same days as the ones recorded now —
    today and days whose footfall is still to come would otherwise make
    every comparison read short."""
    offsets = {(d - start).days for d in logged}
    keep = {prev_start + timedelta(days=o) for o in offsets}
    return _total(_mine(previous)[lambda df: df["Date"].isin(keep)])


mine_now = _mine(current)
sales_days = mine_now.groupby("Date")["Revenue"].sum().loc[lambda s: s > 0].index
foot_days = mine_now.groupby("Date")["Walk-ins"].sum().loc[lambda s: s > 0].index
# Today is still trading, so changes are worked out on completed days only,
# on both sides; the headline figures still include today so far.
done_now = mine_now[mine_now["Date"] < TODAY]
now_done = _total(done_now)
sales_done = [d for d in sales_days if d < TODAY]
foot_done = [d for d in foot_days if d < TODAY]
before_sales, before_foot = _matching(sales_done), _matching(foot_done)
now_done_foot = _total(done_now[done_now["Date"].isin(foot_done)])
foot_only = {"Walk-ins", "Conversion %", "Sales per Walk-in"}


def _was(metric: str):
    base = before_foot if metric in foot_only else before_sales
    if base is None or (metric in foot_only and len(foot_days) == 0):
        return None
    return base[metric]


def _is(metric: str):
    base = now_done_foot if metric in foot_only else now_done
    return base[metric] if base is not None else None


def _delta(metric: str):
    if metric == "Conversion %":
        a, b = _is(metric), _was(metric)
        return a - b if a is not None and b is not None and pd.notna(a) and pd.notna(b) else None
    return _change(_is(metric), _was(metric))


tiles = [
    ("Shop Sales", _kes(now["Revenue"]), _delta("Revenue"), "%"),
    ("Walk-ins", f"{now['Walk-ins']:,.0f}", _delta("Walk-ins"), "%"),
    ("Bought", f"{now['Purchased']:,.0f}", _delta("Purchased"), "%"),
    ("Conversion", _pct(now["Conversion %"]), _delta("Conversion %"), " pts"),
    ("Sales per Walk-in", _kes(now["Sales per Walk-in"]), _delta("Sales per Walk-in"), "%"),
]
only_today = start == end == TODAY
for col, (label, value, delta, unit) in zip(st.columns(len(tiles)), tiles):
    no_footfall = label not in ("Shop Sales", "Bought") and len(foot_days) == 0
    with col.container(border=True):
        st.metric(label, "—" if no_footfall else value,
                  "footfall not counted yet" if no_footfall
                  else "today so far — live" if only_today
                  else f"{delta:+.1f}{unit} vs {prev_name}" if delta is not None
                  else f"no data for {prev_name}",
                  delta_color="normal" if delta is not None and not no_footfall and not only_today
                  else "off")
if not only_today:
    st.caption(f"Changes compare completed days against the same days of the {prev_name}, so "
               "today's trading and days not recorded yet don't make this period look short.")
if pd.notna(now["% of Target"]):
    st.progress(min(now["% of Target"] / 100, 1.0),
                text=f"{now['% of Target']:.0f}% of target (KES {now['Target']:,.0f}, the Odoo "
                     f"daily targets for the days recorded) · {now['Not Purchased']:,.0f} walk-ins "
                     "left without buying")

sources = _mine(current).groupby("Date")["Revenue Source"].first()
live_days = sorted(sources[sources == ff.LIVE].index)
if live_days:
    st.caption("💡 Sales are live POS takings for "
               + ", ".join(f"{d:%a %d %b}" for d in live_days)
               + " — not reconciled by finance yet; every other day is the reconciled figure "
                 "from the Reporting sheet. See **Revenue Check** below for the difference.")
low = _mine(current)[_mine(current)["Footfall Source"].isin([ff.SHEET_COUNT, ff.LOW_COUNT, ff.NOT_COUNTED])
                     & (_mine(current)["Date"] < TODAY)]
if not low.empty:
    missing = int((low["Footfall Source"] == ff.NOT_COUNTED).sum())
    with st.expander(f"Footfall: {len(low)} shop-day(s) not taken from the footfall module"
                     + (f" — {missing} with no count anywhere" if missing else "")):
        st.caption("The footfall module had no count for these, or a count below that day's "
                   "walk-in orders (usually keyed mid-day and not updated at closing). The sheet's "
                   "total is used where it has one; a count below the orders with nothing better "
                   "is raised to the orders. Shop-days with no count anywhere are left out of "
                   "conversion, which is why Bought can be higher than Walk-ins.")
        st.dataframe(pd.DataFrame({
            "Date": low["Date"], "Shop": low["Shop"],
            "Footfall module": low["Footfall"], "Walk-in orders": low["Purchased"].astype(int),
            "Walk-ins used": low["Walk-ins"].astype(int), "From": low["Footfall Source"],
        }).sort_values(["Date", "Shop"], ascending=[False, True]), hide_index=True, width="stretch")

channels = ff.other_channels(rev, system, start, end)
if location == ALL and not channels.empty:
    parts = channels.groupby("Shop")["Revenue"].sum()
    st.caption("Shop sales leave out channels without walk-ins: "
               + " · ".join(f"{name} {_kes(parts.get(name, 0))}" for name in ff.OTHER_CHANNELS)
               + f" — KES {now['Revenue'] + parts.sum():,.0f} with them.")


def _whatsapp_by_shop_day() -> pd.DataFrame | None:
    """WhatsApp chats logged in the WhatsApp Monitor per shop and day, and
    how many bought; None if the monitor can't be read."""
    try:
        wa = db.run_query(queries.WA_INTERACTIONS, {"start_date": start, "end_date": end})
    except Exception:
        return None
    if wa.empty:
        return pd.DataFrame(columns=["Shop", "Date", "Chats", "Converted"])
    frame = pd.DataFrame({
        "Shop": wa["Branch"].replace(ff.SYSTEM_NAMES),
        "Date": pd.to_datetime(wa["Date"]).dt.date,
        "Converted": wa["Converted"].fillna(False).astype(bool),
    })
    return (frame.groupby(["Shop", "Date"])
            .agg(Chats=("Converted", "size"), Converted=("Converted", "sum")).reset_index())


# --- Shop by day: the workbook's "Revenue Sales" sheet -----------------------
def _shop_day_grid() -> None:
    """Every day of the period down the side and every shop across — sales,
    % of its daily target, walk-ins, bought and conversion — with the
    network summary, the channels without walk-ins and the daily total on the
    right, a subtotal per week and a total at the bottom. The layout of the
    weekly "Sales vs Conversion Rate" workbook's Revenue Sales sheet."""
    data = _mine(current)
    if data.empty:
        return
    order = [s for s in shops if s in set(data["Shop"])]
    channel_rows = ff.other_channels(rev, system, start, end) if location == ALL else None
    chats = _whatsapp_by_shop_day()

    def figures(rows: pd.DataFrame, extra: pd.DataFrame | None, on: list) -> dict:
        out = {}
        wa = chats[chats["Date"].isin(on)] if chats is not None else None
        for shop in order:
            r = rows[rows["Shop"] == shop]
            sales, target = r["Revenue"].sum(), r["Target"].sum(min_count=1)
            walk, bought = r["Walk-ins"].sum(), r["Purchased"].sum()
            counted = r["Counted Purchased"].sum()
            out[f"{shop}|Sales"] = sales if not r.empty else None
            out[f"{shop}|Rate"] = sales / target * 100 if target and target > 0 else None
            out[f"{shop}|Walk-ins"] = walk if walk else None
            out[f"{shop}|Bought"] = bought if not r.empty else None
            out[f"{shop}|Conversion"] = counted / walk * 100 if walk else None
            if wa is not None:
                w = wa[wa["Shop"] == shop]
                n, won = w["Chats"].sum(), w["Converted"].sum()
                out[f"{shop}|WA Chats"] = n or None
                out[f"{shop}|WA Bought"] = won if n else None
                out[f"{shop}|WA Conversion"] = won / n * 100 if n else None
        sales, target = rows["Revenue"].sum(), rows["Target"].sum(min_count=1)
        walk, counted = rows["Walk-ins"].sum(), rows["Counted Purchased"].sum()
        out["Total|Sales"] = sales
        out["Total|Rate"] = sales / target * 100 if target and target > 0 else None
        out["Total|Walk-ins"] = walk or None
        out["Total|Bought"] = rows["Purchased"].sum()
        out["Total|Conversion"] = counted / walk * 100 if walk else None
        if wa is not None:
            w = wa[wa["Shop"].isin(order)]
            n, won = w["Chats"].sum(), w["Converted"].sum()
            out["Total|WA Chats"] = n or None
            out["Total|WA Bought"] = won if n else None
            out["Total|WA Conversion"] = won / n * 100 if n else None
        channels_total = 0.0
        for name in ff.OTHER_CHANNELS:
            v = extra[extra["Shop"] == name]["Revenue"].sum() if extra is not None else 0.0
            out[f"Channel|{name}"] = v
            channels_total += v
        out["Daily Total|Sales"] = sales + channels_total
        return out

    lines = []
    days = sorted(data["Date"].unique())
    weeks: dict[int, list] = {}
    for d in days:
        weeks.setdefault(ff.week_number(d), []).append(d)
    pick = lambda frame, ds: frame[frame["Date"].isin(ds)] if frame is not None else None
    for week, week_days in weeks.items():
        for d in week_days:
            lines.append({"Week": f"Week {week}", "Date": f"{d:%a %d %b}",
                          **figures(pick(data, [d]), pick(channel_rows, [d]), [d])})
        if len(weeks) > 1 or len(week_days) > 1:
            lines.append({"Week": f"Week {week} Total", "Date": "",
                          **figures(pick(data, week_days), pick(channel_rows, week_days), week_days)})
    if len(weeks) > 1:
        lines.append({"Week": "Period Total", "Date": "", **figures(data, channel_rows, days)})
    table = pd.DataFrame(lines)

    latest = data.dropna(subset=["Target"]).sort_values("Date").groupby("Shop")["Target"].last()
    groups = [(None, [("Week", "Week", "text")]), (None, [("Date", "Date", "text")])]
    for shop in order:
        head = (f"{shop} · target KES {latest[shop]:,.0f}/day" if shop in latest.index else shop)
        groups.append((head, [(f"{shop}|Sales", "Sales", "money"), (f"{shop}|Rate", "Rate", "pct"),
                              (f"{shop}|Walk-ins", "Walk-ins", "int"),
                              (f"{shop}|Bought", "Bought", "int"),
                              (f"{shop}|Conversion", "Conversion", "pct")]
                       + ([(f"{shop}|WA Chats", "WA Chats", "int"),
                           (f"{shop}|WA Bought", "WA Bought", "int"),
                           (f"{shop}|WA Conversion", "WA Conversion", "pct")] if chats is not None else [])))
    groups.append(("Summary — shops" if location == ALL else f"Summary — {location}",
                   [("Total|Sales", "Total Sales", "money"), ("Total|Rate", "% of Target", "pct"),
                    ("Total|Walk-ins", "Total Walk-ins", "int"), ("Total|Bought", "Bought", "int"),
                    ("Total|Conversion", "Conversion", "pct")]
                   + ([("Total|WA Chats", "WA Chats", "int"), ("Total|WA Bought", "WA Bought", "int"),
                       ("Total|WA Conversion", "WA Conversion", "pct")] if chats is not None else [])))
    if location == ALL:
        groups.append(("No walk-ins", [(f"Channel|{n}", n, "money") for n in ff.OTHER_CHANNELS]))
        groups.append(("Daily Total", [("Daily Total|Sales", "All Sales", "money")]))

    st.subheader("Sales & Conversion by Shop and Day")
    st.caption("The weekly Sales vs Conversion workbook's Revenue Sales sheet: each shop's sales, "
               "the % of its daily target (Rate), walk-ins, how many bought and the conversion, "
               "then its WhatsApp chats (WhatsApp Monitor), how many bought and the WhatsApp "
               "conversion, day by day, with the shops' summary, the channels without walk-ins "
               "and the day's total on the right. Week subtotals and the period total are worked out on "
               "the sums. Scroll sideways for every shop; Date and Week stay put.")
    with st.container(border=True):
        grid.grouped_table(table, groups, pinned=2, bold_rows="Total",
                           height=min(120 + 34 * len(table), 640), key="footfall_shop_day")


_shop_day_grid()


# --- Daily sales vs conversion ------------------------------------------------
daily = ff.summarise(_mine(current), ["Date"]).sort_values("Date")
with st.container(border=True):
    labels = [f"{d:%a %d %b}" for d in daily["Date"]]
    fig = go.Figure()
    fig.add_bar(x=labels, y=daily["Revenue"], name="Sales",
                marker=dict(color=theme.CATEGORICAL[0], cornerradius=4),
                hovertemplate="%{x}<br>Sales <b>KES %{y:,.0f}</b><extra></extra>")
    if daily["Target"].notna().any():
        fig.add_scatter(x=labels, y=daily["Target"], name="Daily target", mode="markers",
                        marker=dict(symbol="line-ew", size=26,
                                    line=dict(width=3, color=theme.STATUS["warning"])),
                        hovertemplate="%{x}<br>Target <b>KES %{y:,.0f}</b><extra></extra>")
    fig.add_scatter(x=labels, y=daily["Conversion %"], name="Conversion rate", yaxis="y2",
                    mode="lines+markers", line=dict(color=theme.CATEGORICAL[4], width=2.5),
                    marker=dict(size=8), connectgaps=False,
                    hovertemplate="%{x}<br>Conversion <b>%{y:.1f}%</b><extra></extra>")
    theme.apply_layout(fig, show_legend=True)
    fig.update_layout(
        title="Daily sales vs conversion rate", height=400, hovermode="x unified",
        yaxis=dict(tickprefix="KES ", tickformat="~s"),
        yaxis2=dict(overlaying="y", side="right", range=[0, 105], ticksuffix="%",
                    showgrid=False, tickfont=dict(color=theme.TEXT_MUTED)),
    )
    theme.show(fig)


# --- Daily summary (the workbook's "Summary Report") --------------------------
st.subheader("Daily Summary")
st.caption("Each day's sales against the summed daily targets, with walk-ins and conversion — "
           "the workbook's Summary Report.")
summary = daily.assign(Week=daily["Date"].map(lambda d: f"Week {ff.week_number(d)}"))
summary = pd.DataFrame({
    "Week": summary["Week"], "Date": summary["Date"],
    "Daily Target": summary["Target"].round(0), "Sales": summary["Revenue"].round(0),
    "% of Target": summary["% of Target"].round(1),
    "Walk-ins": summary["Walk-ins"].astype(int), "Bought": summary["Purchased"].astype(int),
    "Didn't Buy": summary["Not Purchased"].astype(int),
    "Conversion %": summary["Conversion %"].round(1),
    "Sales per Walk-in": summary["Sales per Walk-in"].round(0),
    "Sales from": summary["Date"].map(sources),
}).sort_values("Date", ascending=False)
grid.filterable_table(summary, currency_columns=("Daily Target", "Sales", "Sales per Walk-in"),
                      height=min(120 + 42 * len(summary), 520))


# --- Revenue check: POS takings vs the finance-reconciled sheet ---------------
st.subheader("Revenue Check — POS vs Reconciled")
st.caption("The till's takings as printed at the shop against finance's reconciled revenue on "
           "the Reporting sheet. Variance is reconciled minus POS: what reconciliation added (+) "
           "or took off (−). Days finance hasn't updated yet show as awaiting.")
check = ff.reconciliation(rev, system, start, end)
check_mine = check if location == ALL else check[check["Shop"] == location]
live_mine = _mine(current)
awaiting = (live_mine[live_mine["Revenue Source"] == ff.LIVE]
            .groupby("Date", as_index=False)["System Revenue"].sum())
if location == ALL:
    live_channels = ff.other_channels(rev, system, start, end)
    live_channels = live_channels[live_channels["Date"].isin(set(awaiting["Date"]))]
    extra = live_channels.groupby("Date")["Revenue"].sum()
    awaiting["System Revenue"] += awaiting["Date"].map(extra).fillna(0.0)

if check_mine.empty and awaiting.empty:
    st.info("No revenue to compare in this period.")
else:
    by_day = check_mine.groupby("Date", as_index=False)[["System Revenue", "Sheet Revenue", "Variance"]].sum()
    by_day["Status"] = "Reconciled"
    if not awaiting.empty:
        by_day = pd.concat([by_day, awaiting.assign(**{"Sheet Revenue": None, "Variance": None,
                                                        "Status": "Awaiting reconciliation"})],
                           ignore_index=True)
    by_day["Variance %"] = (by_day["Variance"] / by_day["System Revenue"].where(by_day["System Revenue"] != 0)
                            * 100).round(2)
    pos_total = check_mine["System Revenue"].sum()
    sheet_total = check_mine["Sheet Revenue"].sum()
    if not check_mine.empty:
        c1, c2, c3 = st.columns(3)
        for col, label, value in [(c1, "POS takings (reconciled days)", _kes(pos_total)),
                                  (c2, "Reconciled revenue", _kes(sheet_total)),
                                  (c3, "Variance", f"KES {sheet_total - pos_total:+,.0f}")]:
            with col.container(border=True):
                st.metric(label, value)
    left, right = st.columns([1, 1])
    with left:
        st.caption("By day")
        grid.filterable_table(
            by_day.sort_values("Date", ascending=False).rename(
                columns={"System Revenue": "POS Takings", "Sheet Revenue": "Reconciled"})
            [["Date", "Status", "POS Takings", "Reconciled", "Variance", "Variance %"]],
            currency_columns=("POS Takings", "Reconciled", "Variance"),
            height=min(140 + 42 * len(by_day), 520))
    with right:
        st.caption("By shop and channel, reconciled days only — biggest differences first")
        by_shop_check = check_mine.groupby("Shop", as_index=False).agg(
            **{"POS Takings": ("System Revenue", "sum"), "Reconciled": ("Sheet Revenue", "sum"),
               "Variance": ("Variance", "sum"),
               "Days Off": ("Variance", lambda v: int((v.abs() >= 1).sum()))})
        by_shop_check["Variance %"] = (by_shop_check["Variance"]
                                       / by_shop_check["POS Takings"].where(by_shop_check["POS Takings"] != 0)
                                       * 100).round(2)
        by_shop_check = by_shop_check.reindex(
            by_shop_check["Variance"].abs().sort_values(ascending=False).index)
        grid.filterable_table(by_shop_check, currency_columns=("POS Takings", "Reconciled", "Variance"),
                              pinned_columns=("Shop",), height=min(140 + 42 * len(by_shop_check), 520))


# --- Shops ---------------------------------------------------------------------
st.subheader("Shops")
st.caption(f"Every shop for {period_label}, ranked by conversion rate. Change is in percentage "
           f"points against the {prev_name}.")
by_shop = ff.summarise(current, ["Shop"])
prev_shop = ff.summarise(previous, ["Shop"]).set_index("Shop") if not previous.empty else None
network_rate = (by_shop["Purchased"].sum() / by_shop["Walk-ins"].sum() * 100
                if by_shop["Walk-ins"].sum() else None)

table = pd.DataFrame({
    "Shop": by_shop["Shop"],
    "Target": by_shop["Target"].round(0), "Sales": by_shop["Revenue"].round(0),
    "% of Target": by_shop["% of Target"].round(1),
    "Walk-ins": by_shop["Walk-ins"].astype(int), "Bought": by_shop["Purchased"].astype(int),
    "Didn't Buy": by_shop["Not Purchased"].astype(int),
    "Conversion %": by_shop["Conversion %"].round(1),
    "Change (pts)": [
        round(r["Conversion %"] - prev_shop.at[r["Shop"], "Conversion %"], 1)
        if prev_shop is not None and r["Shop"] in prev_shop.index
        and pd.notna(prev_shop.at[r["Shop"], "Conversion %"]) and pd.notna(r["Conversion %"])
        else None
        for _, r in by_shop.iterrows()],
    "Sales per Walk-in": by_shop["Sales per Walk-in"].round(0),
    "Sales per Buyer": by_shop["Sales per Buyer"].round(0),
}).sort_values("Conversion %", ascending=False, na_position="last")
table.insert(0, "Rank", range(1, len(table) + 1))
grid.filterable_table(table, currency_columns=("Target", "Sales", "Sales per Walk-in", "Sales per Buyer"),
                      pinned_columns=("Shop",), height=min(140 + 42 * len(table), 620))

ranked = by_shop.dropna(subset=["Conversion %"]).sort_values("Conversion %")
if not ranked.empty:
    left, right = st.columns(2)
    with left.container(border=True):
        colors = [theme.CATEGORICAL[0] if s == location or location == ALL else theme.OTHER
                  for s in ranked["Shop"]]
        fig = go.Figure(go.Bar(
            x=ranked["Conversion %"], y=ranked["Shop"], orientation="h",
            marker=dict(color=colors, cornerradius=4),
            customdata=ranked[["Purchased", "Walk-ins"]],
            hovertemplate="%{y}<br><b>%{x:.1f}%</b> · %{customdata[0]:,.0f} of "
                          "%{customdata[1]:,.0f} walk-ins bought<extra></extra>"))
        if network_rate is not None:
            fig.add_vline(x=network_rate, line=dict(color=theme.TEXT_SECONDARY, width=1.5, dash="dot"),
                          annotation_text=f"All shops {network_rate:.1f}%",
                          annotation_font_color=theme.TEXT_SECONDARY, annotation_position="top")
        theme.apply_layout(fig, show_legend=False)
        fig.update_layout(title="Conversion rate by shop", height=max(360, 26 * len(ranked) + 80))
        fig.update_xaxes(ticksuffix="%", range=[0, 105], showgrid=True, gridcolor=theme.GRIDLINE)
        fig.update_yaxes(showgrid=False)
        theme.show(fig)
    with right.container(border=True):
        fig = go.Figure(go.Scatter(
            x=ranked["Walk-ins"], y=ranked["Conversion %"], mode="markers+text",
            text=ranked["Shop"], textposition="top center",
            textfont=dict(color=theme.TEXT_SECONDARY, size=11),
            marker=dict(size=14, color=theme.CATEGORICAL[0], opacity=0.85,
                        line=dict(width=1, color=theme.CHART_SURFACE)),
            customdata=ranked[["Revenue"]],
            hovertemplate="%{text}<br>%{x:,.0f} walk-ins · <b>%{y:.1f}%</b> bought<br>"
                          "Sales KES %{customdata[0]:,.0f}<extra></extra>"))
        if network_rate is not None:
            fig.add_hline(y=network_rate, line=dict(color=theme.TEXT_SECONDARY, width=1.5, dash="dot"))
        theme.apply_layout(fig, show_legend=False)
        fig.update_layout(title="Traffic vs conversion — busy shops below the line lose the most",
                          height=max(360, 26 * len(ranked) + 80), hovermode="closest")
        fig.update_xaxes(title="Walk-ins", showgrid=True, gridcolor=theme.GRIDLINE)
        fig.update_yaxes(title="Conversion", ticksuffix="%")
        theme.show(fig)


# --- Shop x day conversion (the workbook's "Walk in Tracker") -----------------
if days <= 31 and not current.empty:
    st.subheader("Conversion by Shop and Day")
    st.caption("Each cell is the share of that day's walk-ins who bought; blank means no "
               "footfall logged. Hover for the counts.")
    grid_days = sorted(current["Date"].unique())
    cells = current.set_index(["Shop", "Date"])
    order = [s for s in shops if s in set(current["Shop"])]
    z, text, hover = [], [], []
    for shop in order:
        zr, tr, hr = [], [], []
        for d in grid_days:
            row = cells.loc[(shop, d)] if (shop, d) in cells.index else None
            walk = row["Walk-ins"] if row is not None else 0
            rate = row["Purchased"] / walk * 100 if walk else None
            zr.append(rate)
            tr.append(f"{rate:.0f}%" if rate is not None else "")
            hr.append(f"{row['Purchased']:,.0f} of {walk:,.0f} bought" if walk else "No footfall logged")
        z.append(zr); text.append(tr); hover.append(hr)
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{d:%a %d %b}" for d in grid_days], y=order, text=text, customdata=hover,
        texttemplate="%{text}", textfont=dict(size=11), zmin=0, zmax=100,
        colorscale=[[i / (len(theme.SEQUENTIAL_BLUE) - 1), c]
                    for i, c in enumerate(theme.SEQUENTIAL_BLUE)],
        colorbar=dict(ticksuffix="%", thickness=10, tickfont=dict(color=theme.TEXT_MUTED)),
        hovertemplate="%{y} · %{x}<br><b>%{z:.1f}%</b> — %{customdata}<extra></extra>",
        xgap=2, ygap=2))
    theme.apply_layout(fig, show_legend=False)
    fig.update_layout(height=max(380, 30 * len(order) + 90), hovermode="closest")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    with st.container(border=True):
        theme.show(fig)


# --- Weekly conversion per shop (the workbook's per-shop week table) ---------
st.subheader(f"Weekly Conversion — last {WEEKS_SHOWN} weeks")
st.caption("Weeks run Sunday to Saturday and are numbered as in the workbook (4–10 Oct 2026 is "
           "Week 41). The current week is still running. This table ignores the period picker.")
if recent.empty:
    st.info("No footfall in the last few weeks.")
else:
    wk = recent.assign(Week=recent["Date"].map(ff.week_start))
    weekly = ff.summarise(wk, ["Shop", "Week"])
    network = ff.summarise(wk, ["Week"]).assign(Shop="All Shops")
    weekly = pd.concat([weekly, network], ignore_index=True)
    weeks = sorted(weekly["Week"].unique())
    names = {w: f"Wk {ff.week_number(w)} ({w:%d %b})" + (" · to date" if w == ff.week_start(TODAY) else "")
             for w in weeks}
    metric = st.radio("Show", ["Conversion %", "Walk-ins", "Bought", "% of Target"],
                      horizontal=True, key="footfall_week_metric")
    column = "Purchased" if metric == "Bought" else metric
    pivot = (weekly.pivot(index="Shop", columns="Week", values=column)
             .reindex([s for s in shops if s in set(weekly["Shop"])] + ["All Shops"])
             .rename(columns=names))
    pct = metric.endswith("%")
    fmt = "%.1f%%" if pct else "%d"
    st.dataframe(
        pivot, width="stretch", height=min(60 + 35 * len(pivot), 760),
        column_config={c: st.column_config.NumberColumn(c, format=fmt) for c in pivot.columns},
    )

st.caption(f"Last updated {datetime.now().strftime('%H:%M:%S')} · the sheet is re-read every "
           "10 minutes, or straight away with Refresh now")
