"""Shop Attendants — who served and who ran the till, for every POS session."""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from lib import db, feedback, filters, footfall, grid, queries, report, scorecards, targets, theme, training

st.set_page_config(page_title="Shop Attendants · Denri Sales Operations Dashboard",
                   page_icon="🧑‍💼", layout="wide")


st.title("🧑‍💼 Shop Attendants")
st.caption("Every POS session with its shop attendants, brand ambassadors and cashier — "
           "live from Postgres")

connected, detail = db.check_connection()
if not connected:
    st.error(f"Could not connect to Postgres: {detail}", icon="🚫")
    st.stop()

col_picker, col_refresh = st.columns([3, 1])
with col_picker:
    start_date, end_date = filters.date_range_control("attendants", default="Yesterday")
with col_refresh:
    db.refresh_button(key="attendants_refresh")

# The headers asked for, in the order asked, with the brand ambassador beside the
# attendant; Session and Orders follow so a row can be traced back in Odoo.
COLUMNS = ["Date", "Location", "Sales Made", "Shop Attendant", "Brand Ambassador (BA)",
           "Cashier", "Session", "Orders"]
NOT_RECORDED = "Not recorded"
BA_SUFFIX = " (BA)"
ALL_PEOPLE = "All Shop Attendants & BAs"


def _served_by(df: pd.DataFrame) -> pd.Series:
    """Who made each row's sales: the attendant, or the BA labelled as one.

    An order carries one or the other, never both.
    """
    ba = df["Brand Ambassador (BA)"].fillna("")
    return df["Shop Attendant"].where(ba == "", ba + BA_SUFFIX)


@st.fragment()
def render_attendants(start_date: date, end_date: date) -> None:
    df = db.run_query(queries.SHOP_ATTENDANT_SESSIONS,
                      {"start_date": start_date, "end_date": end_date})
    if df.empty:
        st.info("No POS sessions with sales in this date range.")
        return
    df = df.assign(**{"Served By": _served_by(df)})

    col_loc, col_person = st.columns(2)
    with col_loc:
        location = st.selectbox("Location", ["All Locations"] + sorted(df["Location"].unique()),
                                key="attendants_location")
    shown = df if location == "All Locations" else df[df["Location"] == location]
    with col_person:
        people = sorted(p for p in shown["Served By"].unique() if p != NOT_RECORDED)
        person = st.selectbox("Shop Attendant / BA", [ALL_PEOPLE] + people,
                              key="attendants_person")
    shown = shown if person == ALL_PEOPLE else shown[shown["Served By"] == person]

    unrecorded = shown[shown["Served By"] == NOT_RECORDED]
    bas = shown[shown["Brand Ambassador (BA)"] != ""]
    items = [
        ("Sales Made", f"KES {shown['Sales Made'].sum():,.0f}", None),
        ("POS Sessions", f"{shown['Session'].nunique():,}", f"{shown['Location'].nunique()} locations"),
        ("Attendants & BAs", f"{len(set(shown['Served By']) - {NOT_RECORDED}):,}",
         f"{bas['Brand Ambassador (BA)'].nunique()} BAs, KES {bas['Sales Made'].sum():,.0f}"),
        ("Nobody Recorded", f"KES {unrecorded['Sales Made'].sum():,.0f}",
         f"{int(unrecorded['Orders'].sum()):,} orders"),
    ]
    for col, (label, value, note) in zip(st.columns(len(items)), items):
        with col.container(border=True):
            st.metric(label, value, note, delta_color="off")

    st.caption(
        f"Last updated {datetime.now().strftime('%H:%M:%S')} · one row per session for each "
        "shop attendant or brand ambassador who served in it — an order carries one or the "
        "other. Cashier is the person who opened the session in Odoo; Sales Made is counted "
        "like the Sales page (delivery fees out, refunds netted, in KES)."
    )

    # One section at a time: the feedback section reads the Google Sheet and
    # every converted customer, which is slow, so it only runs when opened.
    sections = ["Sessions", "Score Cards", "Sales Training", "Customer Feedback"]
    if person != ALL_PEOPLE:
        sections.insert(1, "Sales Breakdown")
    section = st.segmented_control("Section", sections, default=sections[0],
                                   key="attendants_section", label_visibility="collapsed")
    section = section if section in sections else sections[0]

    per, basis = _individual_targets(df, shown, start_date, end_date)
    _export_report(df, shown, per, basis, start_date, end_date, location, person)

    if section == "Sessions":
        if person == ALL_PEOPLE:
            _by_person_chart(per, basis)
        with st.container(border=True):
            st.caption(f"{len(shown):,} rows · click a column header's filter icon to search or "
                       "narrow it. The last row is the total for the range.")
            sessions = shown[COLUMNS].copy()
            total = {c: "" for c in COLUMNS}
            total.update({"Date": "TOTAL", "Sales Made": round(sessions["Sales Made"].sum(), 2),
                          "Session": f"{shown['Session'].nunique():,} sessions",
                          "Orders": int(sessions["Orders"].sum())})
            grid.filterable_table(pd.concat([sessions, pd.DataFrame([total])], ignore_index=True),
                                  currency_columns=("Sales Made",), height=440)

        st.subheader("By Shop Attendant / BA")
        with st.container(border=True):
            st.caption("Each attendant's and BA's sales against their individual target (their "
                       "shop's target shared equally between everyone present there in the "
                       "period), ranked by achievement, with the total at the bottom. Sales "
                       "nobody was recorded on are left out.")
            table = _ranking_table(per)
            sales, target = table["Sales Made"].sum(), table["Individual Target"].sum()
            targeted = per["Targeted"].sum()
            total = {"Rank": None, "Shop Attendant / BA": "TOTAL",
                     "Locations": f"{shown['Location'].nunique()} locations",
                     "Team Size": "", "Sessions": int(table["Sessions"].sum()),
                     "Orders": int(table["Orders"].sum()), "Sales Made": round(sales, 2),
                     "Individual Target": round(target, 0),
                     "Achievement %": round(targeted / target * 100, 1) if target else None,
                     "Share %": 100.0,
                     "Avg Sale": round(sales / max(table["Orders"].sum(), 1), 0)}
            grid.filterable_table(pd.concat([table, pd.DataFrame([total])], ignore_index=True),
                                  currency_columns=("Sales Made", "Individual Target", "Avg Sale"),
                                  pinned_columns=("Shop Attendant / BA",), height=440)
    elif section == "Sales Training":
        _training(start_date, end_date, location)
    elif section == "Score Cards":
        _score_cards(start_date, end_date, location, person)
    elif section == "Sales Breakdown":
        _person_sales(person, start_date, end_date, location)
    else:
        _feedback_table(start_date, end_date, location, person)


def _individual_targets(df: pd.DataFrame, shown: pd.DataFrame,
                        start_date: date, end_date: date) -> tuple[pd.DataFrame, targets.Basis]:
    """Each attendant's / BA's sales against their own target.

    A person's target is their shop's target for the period shared equally
    between everyone present at that shop in it — the attendants and BAs
    recorded on its orders (Odoo's attendance check-in is not in use yet).
    Team sizes come from the whole range at every shop (df), so picking a
    location or person never changes anyone's target. Someone who worked at
    two shops carries a share of both, and only their sales at shops with a
    target count towards Achievement.
    """
    rows = db.run_query(queries.SALES_TARGET_ROWS, {"start_date": start_date, "end_date": end_date})
    basis = targets.basis_for(start_date, end_date, rows)
    team = df[df["Served By"] != NOT_RECORDED].groupby("Location")["Served By"].nunique()
    pair = (shown[shown["Served By"] != NOT_RECORDED]
            .groupby(["Served By", "Location"], as_index=False)
            .agg(Sales=("Sales Made", "sum"), Orders=("Orders", "sum"),
                 Sessions=("Session", "nunique")))
    pair["Team Size"] = pair["Location"].map(team)
    pair["Target"] = pair["Location"].map(pd.Series(basis.targets, dtype=float)) / pair["Team Size"]
    pair["Targeted Sales"] = pair["Sales"].where(pair["Target"].notna(), 0.0)
    per = pair.groupby("Served By", as_index=False).agg(
        Locations=("Location", lambda v: ", ".join(sorted(set(v)))),
        Team=("Team Size", lambda v: ", ".join(str(int(n)) for n in v)),
        Sessions=("Sessions", "sum"), Orders=("Orders", "sum"), Sales=("Sales", "sum"),
        Target=("Target", lambda v: v.sum(min_count=1)),
        Targeted=("Targeted Sales", "sum"))
    per["Achievement %"] = (per["Targeted"] / per["Target"].where(per["Target"] > 0) * 100).round(1)
    per = per.sort_values(["Achievement %", "Sales"], ascending=[False, False], na_position="last")
    return per.reset_index(drop=True), basis


def _by_person_chart(per: pd.DataFrame, basis: targets.Basis) -> None:
    ranked = per.dropna(subset=["Achievement %"]).sort_values("Achievement %")
    if ranked.empty:
        return
    due = basis.expected * 100

    def status(pct: float) -> str:
        if pct >= due:
            return theme.STATUS["good"]
        return theme.STATUS["warning"] if pct >= due * 0.8 else theme.STATUS["critical"]

    with st.container(border=True):
        fig = go.Figure()
        fig.add_bar(
            y=ranked["Served By"], x=ranked["Achievement %"], orientation="h",
            marker=dict(color=[status(p) for p in ranked["Achievement %"]], cornerradius=4),
            text=[f"{p:.0f}%" for p in ranked["Achievement %"]], textposition="outside",
            textfont=dict(color=theme.TEXT_SECONDARY, size=11), cliponaxis=False,
            customdata=ranked[["Sales", "Target", "Locations", "Team"]],
            hovertemplate=("<b>%{y}</b> — %{customdata[2]}<br><b>%{x:.1f}%</b> of target"
                           "<br>Sales KES %{customdata[0]:,.0f} of KES %{customdata[1]:,.0f}"
                           "<br>Team of %{customdata[3]} sharing the shop target<extra></extra>"),
        )
        fig.add_vline(x=due, line=dict(color=theme.TEXT_SECONDARY, width=1.5, dash="dot"),
                      annotation_text="Target" if due >= 100 else f"Due by now ({due:.0f}%)",
                      annotation_font_color=theme.TEXT_SECONDARY, annotation_position="top")
        theme.apply_layout(fig, show_legend=False)
        fig.update_layout(title="Shop Attendants / BAs Ranked by Achievement Against Individual Target",
                          hovermode="closest", height=max(380, 26 * len(ranked) + 90))
        top = max(ranked["Achievement %"].max(), due) * 1.12
        fig.update_xaxes(showgrid=True, gridcolor=theme.GRIDLINE, ticksuffix="%", range=[0, top])
        fig.update_yaxes(showgrid=False)
        theme.show(fig)
        missing = per[per["Achievement %"].isna()]["Served By"].tolist()
        st.caption(
            f"Individual target = the shop's {basis.label} shared equally between the attendants "
            "and BAs present at that shop in the period (recorded on its orders). Green is at or "
            f"above {'target' if due >= 100 else 'what is due by now'}, amber within 20% of it, "
            "red further behind."
            + (f" Not ranked (no shop target): {', '.join(missing)}." if missing else ""))


def _ranking_table(per: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame({
        "Shop Attendant / BA": per["Served By"], "Locations": per["Locations"],
        "Team Size": per["Team"], "Sessions": per["Sessions"], "Orders": per["Orders"].astype(int),
        "Sales Made": per["Sales"].round(2), "Individual Target": per["Target"].round(0),
        "Achievement %": per["Achievement %"],
        "Share %": (per["Sales"] / per["Sales"].sum() * 100).round(1),
        "Avg Sale": (per["Sales"] / per["Orders"].where(per["Orders"] > 0)).round(0),
    })
    table.insert(0, "Rank", [i + 1 if pd.notna(a) else None
                             for i, a in enumerate(table["Achievement %"])])
    return table


SCORE_SHOP_COLUMNS = (["Location", "Head", "Score", "Weight Covered %"] + scorecards.HEAD_SCORE_COLUMNS
                      + ["Revenue", "Target", "Target %", "Orders", "Basket Value", "Basket Value %",
                         "Walk-ins", "Walk-in Orders", "Walk-in Conversion %",
                         "WhatsApp Chats", "WhatsApp Converted", "WhatsApp Conversion %",
                         "Chats", "Response Time (min)", "Customers", "Feedback", "Feedback %",
                         "Forms Received", "Forms Matched",
                         "Training Completion %", "Training %", "Stock Variance %",
                         "Last Stock Count", "Attendants", "Revenue per Attendant", "Bags",
                         "Basket Size", "Repeat %", "Retention %", "Corporate Revenue"])
SCORE_PEOPLE_COLUMNS = (["Attendant / BA", "Location", "Score", "Weight Covered %"]
                        + scorecards.ATTENDANT_SCORE_COLUMNS
                        + ["Days Worked", "Avg Team", "Target", "Revenue", "Target %",
                           "Bulk Revenue", "Bulk Target", "Bulk %", "Orders", "Basket Value",
                           "Basket Value %", "New Bag Revenue", "New Products %",
                           "Customers Served", "Walk-in Orders", "Conversion %",
                           "WhatsApp Days", "Chats Answered", "Fast Reply %",
                           "Service Rating", "Ratings", "Customers", "Feedback", "Feedback %",
                           "Captured Orders", "Capture %", "Training %", "Attendance %",
                           "Late Days", "Revenue per Day", "Bags", "Basket Size",
                           "Corporate Revenue"])


def _export_report(df: pd.DataFrame, shown: pd.DataFrame, per: pd.DataFrame,
                   basis: targets.Basis, start_date: date, end_date: date,
                   location: str, person: str) -> None:
    """Prepare the page's stats as an Excel workbook and a Word document.

    Built only when asked for (the score cards read the feedback and training
    sheets, which is slow) and kept for the current selection, so the two
    download buttons don't rebuild it."""
    key = (start_date, end_date, location, person)
    with st.expander("📄 Export report — Excel (.xlsx) and Word (.docx)"):
        st.caption("Summary, ranking against individual targets, shop and attendant score "
                   "cards, and every session (Excel only) for the period and filters chosen above.")
        if st.button("Prepare report", key="attendants_report_prepare", type="primary"):
            with st.spinner("Preparing the report…"):
                rep = _build_report(df, shown, per, basis, start_date, end_date, location, person)
                st.session_state["attendants_report"] = (key, report.to_xlsx(rep), report.to_docx(rep))
        saved = st.session_state.get("attendants_report")
        if saved and saved[0] == key:
            stem = f"Shop_Attendants_{start_date:%Y-%m-%d}" + (
                "" if start_date == end_date else f"_to_{end_date:%Y-%m-%d}")
            if location != "All Locations":
                stem += f"_{location}"
            if person != ALL_PEOPLE:
                stem += "_" + "".join(ch if ch.isalnum() else "_" for ch in person).strip("_")
            c1, c2 = st.columns(2)
            c1.download_button("⬇️ Download Excel (.xlsx)", saved[1], file_name=f"{stem}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               key="attendants_report_xlsx", width="stretch")
            c2.download_button("⬇️ Download Word (.docx)", saved[2], file_name=f"{stem}.docx",
                               mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                               key="attendants_report_docx", width="stretch")
        elif saved:
            st.caption("The filters changed since the last report — prepare it again.")


def _build_report(df: pd.DataFrame, shown: pd.DataFrame, per: pd.DataFrame,
                  basis: targets.Basis, start_date: date, end_date: date,
                  location: str, person: str) -> report.Report:
    period = (f"{start_date:%A %d %B %Y}" if start_date == end_date
              else f"{start_date:%d %b %Y} – {end_date:%d %b %Y}")
    unrecorded = shown[shown["Served By"] == NOT_RECORDED]
    ranking = _ranking_table(per)
    on_target = ranking["Achievement %"].ge(basis.expected * 100).sum()
    summary = [
        ("Period", period), ("Location", location), ("Shop Attendant / BA", person),
        ("Sales Made", f"KES {shown['Sales Made'].sum():,.0f}"),
        ("POS Sessions", f"{shown['Session'].nunique():,} at {shown['Location'].nunique()} locations"),
        ("Attendants & BAs", f"{len(set(shown['Served By']) - {NOT_RECORDED}):,}"),
        ("Sales with nobody recorded", f"KES {unrecorded['Sales Made'].sum():,.0f} "
                                       f"({int(unrecorded['Orders'].sum()):,} orders)"),
        ("Target basis", f"{basis.label} · {basis.expected * 100:.0f}% due by now"),
        ("At or above target", f"{on_target} of {ranking['Achievement %'].notna().sum()} ranked"),
    ]
    tables = [report.Table(
        "Ranking vs Individual Target", ranking,
        note="Individual target = the shop's target for the period shared equally between the "
             "attendants and BAs present at that shop in it. Ranked by achievement.")]

    data = _scorecard_data(start_date, end_date)
    notes = []
    if data is not None:
        shops, people, overall = data["shops"], data["people"], data["shop_cards"]
        if location != "All Locations":
            shops = shops[shops["Location"] == location]
            overall = overall[overall["Location"] == location]
            people = people[people["Location"] == location]
        if person != ALL_PEOPLE:
            people = people[people["Attendant / BA"] == person]
        tables += [
            report.Table("Shop Score Cards", overall[SHOP_CARD_COLUMNS],
                         note="The shop as a whole; every measure counts equally.",
                         doc_columns=["Location", "Score"] + scorecards.SHOP_SCORE_COLUMNS),
            report.Table("Shop Manager Score Cards", shops[SCORE_SHOP_COLUMNS],
                         note="Weighted score — " + _weights_text(scorecards.HEAD_MEASURES)
                              + ". Each measure scored out of 100 against its target.",
                         doc_columns=["Location", "Head", "Score"] + scorecards.HEAD_SCORE_COLUMNS),
            report.Table("Attendant Score Cards", people[SCORE_PEOPLE_COLUMNS],
                         note="Weighted score — " + _weights_text(scorecards.ATTENDANT_MEASURES)
                              + ". Measures a person had no part in are left out and the rest "
                                "scaled up to 100% (Weight Covered %).",
                         doc_columns=["Attendant / BA", "Location", "Score", "Weight Covered %",
                                      "Personal Target Score", "Basket Score", "Conversion Score",
                                      "Respond Fast Score", "Service Score", "Training Score"]),
        ]
        notes = [n for n in (data["feedback_note"], data["training_note"]) if n]
    tables.append(report.Table("Sessions", shown[COLUMNS].sort_values(["Date", "Location"]),
                               in_doc=False))
    title = "Shop Attendants Report"
    return report.Report(title, f"{period} · {location} · {person}", summary, tables, notes)


def _feedback_table(start_date: date, end_date: date, location: str, person: str) -> None:
    """How many of each person's customers left feedback about their purchase."""
    st.subheader("Customer Feedback by Shop Attendant / BA")
    try:
        responses = feedback.load()
    except feedback.FeedbackUnavailable as exc:
        st.warning(
            f"{exc} The feedback lives in the Google Sheet the POS Request Hub reads "
            f"(\"{feedback.WORKSHEET}\" tab). Share that sheet with "
            f"**{feedback.service_account_email()}** as a Viewer and this table fills in "
            "on the next refresh.",
            icon="🔒",
        )
        return

    orders = db.run_query(queries.SCORECARD_ORDERS,
                          {"start_date": start_date, "end_date": end_date})
    if orders.empty:
        st.info("No sales in this range.")
        return
    not_customers = scorecards.placeholders(orders)
    df = scorecards.apply_feedback(scorecards.prepare(orders, not_customers),
                                   _match_forms(orders, not_customers, end_date, responses))
    if location != "All Locations":
        df = df[df["Location"] == location]
    if person != ALL_PEOPLE:
        df = df[df["Served By"] == person]
    df = df[df["Served By"] != NOT_RECORDED]
    if df.empty:
        st.info("No sales with an attendant or BA recorded in this range.")
        return

    # One customer per person, however many shops or orders they bought
    # across; an order with no usable phone (the shop's own number typed in)
    # is a customer of its own, so a form matched to it by name still counts.
    df = df.assign(Key=df["Customer"].where(df["Customer"].notna(),
                                            "order:" + df["Order ID"].astype(str)))
    per_customer = (df.groupby(["Served By", "Key"], as_index=False)
                    .agg(Customer=("Customer Name", "first"), Phone=("Customer", "first"),
                         Locations=("Location", lambda v: ", ".join(sorted(set(v)))),
                         Bought=("Day", "min"), Orders=("Order ID", "size"),
                         **{"Gave Feedback": ("Feedback", "max"),
                            "Feedback On": ("Feedback On", "min"),
                            "Matched By": ("Feedback Match", "first")}))
    per_customer["Gave Feedback"] = per_customer["Gave Feedback"].fillna(False).astype(bool)
    per_customer["Feedback On"] = pd.to_datetime(per_customer["Feedback On"]).dt.date

    table = (per_customer.groupby("Served By")
             .agg(Location=("Locations", lambda v: ", ".join(sorted({x for s in v for x in s.split(", ")}))),
                  **{"Customers Served": ("Key", "size"),
                     "Gave Feedback": ("Gave Feedback", "sum"),
                     "By Phone": ("Matched By", lambda v: int((v == "Phone").sum())),
                     "By Near-miss Phone / Name": ("Matched By", lambda v: int(v.isin(["Near-miss phone", "Name"]).sum()))})
             .reset_index().rename(columns={"Served By": "Shop Attendant / BA"}))
    table["Feedback Rate"] = (table["Gave Feedback"] / table["Customers Served"] * 100).round(1)
    table = table.sort_values(["Feedback Rate", "Customers Served"], ascending=[False, False])

    day = responses["Submitted"].dt.date
    window = responses[(day >= start_date) & (day <= end_date)]
    if location != "All Locations":
        window = window[window["Branch"].astype(str).str.strip().str.lower() == location.lower()]
    served, gave = int(table["Customers Served"].sum()), int(table["Gave Feedback"].sum())
    items = [
        ("Customers Served", f"{served:,}", None),
        ("Gave Feedback", f"{gave:,}", f"{gave / served * 100:.1f}% of customers served" if served else None),
        ("Forms Received", f"{len(window):,}",
         "naming this shop in this range" if location != "All Locations" else "all forms in this range"),
    ]
    for col, (label, value, note) in zip(st.columns(len(items)), items):
        with col.container(border=True):
            st.metric(label, value, note, delta_color="off")
    st.caption(
        "A customer counts as giving feedback when a form is matched to their purchase in this "
        "range: by the phone number (last nine digits) — the form can come in any time up to "
        f"{scorecards.FEEDBACK_WINDOW_DAYS} days after, even after the range — or, when the "
        "number on the form isn't the one on the POS, by the first name on the form against "
        f"the shop's customers in the {scorecards.NAME_MATCH_DAYS} days before, or a number "
        "one or two digits off theirs (a typo). Each customer counts once per attendant or BA. Forms Received can still be higher: forms "
        "about purchases before the range, and forms that couldn't be matched at all."
    )
    with st.container(border=True):
        grid.filterable_table(table, pinned_columns=("Shop Attendant / BA",), height=420)

    if person != ALL_PEOPLE:
        # A section, not an expander: the grid draws inside an iframe, and in a
        # collapsed expander it sizes itself to nothing and stays blank when
        # the expander is opened.
        st.markdown(f"**{person}'s customers and their feedback**")
        detail = per_customer[["Customer", "Phone", "Locations", "Bought", "Orders",
                               "Gave Feedback", "Feedback On", "Matched By"]].copy()
        detail["Phone"] = detail["Phone"].where(~detail["Phone"].astype(str).str.startswith("partner:"))
        detail["Gave Feedback"] = detail["Gave Feedback"].map({True: "Yes", False: "No"})
        with st.container(border=True):
            grid.filterable_table(detail.sort_values(["Gave Feedback", "Bought"],
                                                     ascending=[False, False]),
                                  pinned_columns=("Customer",), height=420)


def _person_sales(person: str, start_date: date, end_date: date, location: str) -> None:
    """Everything one attendant or BA sold: customer, number and products."""
    name = person[: -len(BA_SUFFIX)] if person.endswith(BA_SUFFIX) else person
    st.subheader(f"Sales by {person}")
    sales = db.run_query(queries.SHOP_ATTENDANT_SALES,
                         {"start_date": start_date, "end_date": end_date, "person": name})
    if location != "All Locations":
        sales = sales[sales["Location"] == location]
    if sales.empty:
        st.info("No product sales recorded for this person in this range.")
        return

    orders = {o.strip() for refs in sales["Orders"] for o in str(refs).split(",")}
    named = sales[sales["Customer"] != "Walk-in (no name)"]
    items = [
        ("Sales Converted", f"{len(orders):,}", "orders"),
        ("Customers", f"{named['Customer'].nunique():,}",
         f"{len(sales) - len(named):,} lines with no customer name"),
        ("Units Sold", f"{sales['Quantity'].sum():,.0f}", None),
        ("Value", f"KES {sales['Total'].sum():,.0f}", None),
    ]
    for col, (label, value, note) in zip(st.columns(len(items)), items):
        with col.container(border=True):
            st.metric(label, value, note, delta_color="off")

    st.caption(
        "One line per customer and product each day — several units of the same bag read as "
        "one line with its quantity. A combo shows as its own line with the price, beside the "
        "bags inside it at no charge. Refunded items are netted out; delivery fees and order "
        "discounts are not products, so they are left out."
    )
    with st.container(border=True):
        grid.filterable_table(
            sales[["Date", "Customer", "Phone", "Product", "Quantity", "Total", "Location",
                   "Orders"]],
            currency_columns=("Total",), pinned_columns=("Customer",), height=480)


def _training(start_date: date, end_date: date, location: str) -> None:
    """Daily sales training: who answered the questionnaire, how fully and how well."""
    try:
        data = training.load()
    except training.TrainingUnavailable as exc:
        st.warning(str(exc), icon="🔒")
        return
    roster, responses = data["roster"], data["responses"]
    today = date.today()
    days = [d for d in data["days"] if start_date <= d <= end_date]
    closed = [d for d in days if d < today]
    if not days:
        st.info("No questionnaire was given in this date range. Questionnaires on record: "
                + (", ".join(f"{d:%d %b}" for d in data["days"]) or "none") + ".")
        return

    team = roster if location == "All Locations" else roster[roster["Location"] == location]
    answers = responses[responses["Day"].isin(days)] if not responses.empty else responses
    if location != "All Locations" and not answers.empty:
        answers = answers[answers["Location"] == location]
    grid_rows = training.due(data, closed)
    if location != "All Locations":
        grid_rows = grid_rows[grid_rows["Location"] == location]

    expected = len(grid_rows)
    answered = int(grid_rows["Answered"].sum()) if expected else 0
    done = grid_rows[grid_rows["Answered"]] if expected else grid_rows
    latest = max(closed) if closed else None
    missed_latest = (grid_rows[(grid_rows["Day"] == latest) & ~grid_rows["Answered"]]
                     if latest else grid_rows.iloc[0:0])
    _kpis_row([
        ("Response Rate", f"{answered / expected * 100:.1f}%" if expected else "—",
         f"{answered:,} of {expected:,} answers due"),
        ("Answered Same Day", f"{done['On Time'].mean() * 100:.1f}%"
         if not done.empty else "—", "of the answers given"),
        ("Completion Rate", f"{done['Completion %'].mean():.1f}%"
         if not done.empty else "—", "questions answered"),
        ("Average Score", f"{done['Score %'].mean():.1f}%"
         if not done.empty else "—", f"{len(closed)} questionnaires"),
        (f"Missed {latest:%d %b}" if latest else "Missed Latest",
         f"{len(missed_latest):,}", f"of {len(team):,} on the team list"),
    ])
    notes = [f"{len(days)} questionnaire(s) in this range"]
    if today in days:
        notes.append(f"today's ({today:%d %b}) is still open and not counted yet")
    if data["skipped"]:
        notes.append("left out: " + "; ".join(data["skipped"]))
    st.caption(
        " · ".join(notes) + ". Everyone on the team list is expected to answer every "
        "questionnaire. An answer counts for the day of its questionnaire whenever it was "
        "sent; **Answered Same Day** is the share sent that day. **Completion** is the share "
        "of questions answered; **Score** is the form's own mark.")
    if not expected:
        return

    view = st.segmented_control("Training view", ["🏪 By Shop", "✅ Tracker", "🧑‍💼 Respondents"],
                                default="🏪 By Shop", key="training_view",
                                label_visibility="collapsed") or "🏪 By Shop"
    grain = st.radio("Group by", ["Day", "Week (Sun–Sat)", "Month"], horizontal=True,
                     key="training_grain")
    period = pd.to_datetime(grid_rows["Day"])
    if grain == "Day":
        grid_rows["Period"] = period.dt.strftime("%a %d %b")
        order = sorted(set(zip(period, grid_rows["Period"])))
    elif grain == "Month":
        grid_rows["Period"] = period.dt.strftime("%B %Y")
        order = sorted(set(zip(period.dt.to_period("M").dt.start_time, grid_rows["Period"])))
    else:
        sunday = period - pd.to_timedelta((period.dt.weekday + 1) % 7, unit="D")
        grid_rows["Period"] = ("Week " + sunday.dt.strftime("%d %b") + " – "
                               + (sunday + pd.Timedelta(days=6)).dt.strftime("%d %b"))
        order = sorted(set(zip(sunday, grid_rows["Period"])))
    periods = list(dict.fromkeys(label for _, label in order))

    def summary(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
        g = frame.groupby(by)
        out = pd.DataFrame({
            "Team": g["Email"].nunique(),
            "Answers Due": g.size(),
            "Answered": g["Answered"].sum().astype(int),
        })
        out["Missed"] = out["Answers Due"] - out["Answered"]
        out["Response Rate %"] = (out["Answered"] / out["Answers Due"] * 100).round(1)
        ans = frame[frame["Answered"]].groupby(by)
        out["Same Day %"] = (ans["On Time"].mean() * 100).round(1)
        out["Completion %"] = ans["Completion %"].mean().round(1)
        out["Average Score %"] = ans["Score %"].mean().round(1)
        return out

    if view == "🏪 By Shop":
        shops = summary(grid_rows, ["Location"]).reset_index().sort_values("Response Rate %",
                                                                           ascending=False)
        with st.container(border=True):
            ranked = shops.sort_values("Response Rate %")
            fig = go.Figure()
            fig.add_bar(y=ranked["Location"], x=ranked["Response Rate %"], orientation="h",
                        name="Response rate", marker=dict(color=theme.CATEGORICAL[0], cornerradius=4),
                        customdata=ranked[["Answered", "Answers Due"]],
                        hovertemplate="<b>%{y}</b><br>%{x:.1f}% · %{customdata[0]} of "
                                      "%{customdata[1]}<extra></extra>")
            fig.add_bar(y=ranked["Location"], x=ranked["Average Score %"], orientation="h",
                        name="Average score", marker=dict(color=theme.CATEGORICAL[1], cornerradius=4),
                        hovertemplate="<b>%{y}</b><br>Average score %{x:.1f}%<extra></extra>")
            theme.apply_layout(fig, show_legend=True)
            fig.update_layout(title="Response Rate and Average Score by Shop", barmode="group",
                              hovermode="closest", height=max(420, 40 * len(ranked) + 100))
            fig.update_xaxes(range=[0, 105], ticksuffix="%")
            theme.show(fig)
        with st.container(border=True):
            st.caption("Each shop over the range, with the total at the bottom.")
            grid.filterable_table(shops, pinned_columns=("Location",), height=420)

        trend = summary(grid_rows, ["Period"]).reindex(periods).reset_index()
        with st.container(border=True):
            fig = go.Figure()
            for col, color in (("Response Rate %", theme.CATEGORICAL[0]),
                               ("Completion %", theme.CATEGORICAL[3]),
                               ("Average Score %", theme.CATEGORICAL[1])):
                fig.add_scatter(x=trend["Period"], y=trend[col], name=col.replace(" %", ""),
                                mode="lines+markers", line=dict(color=color, width=2),
                                hovertemplate=f"%{{x}}<br>{col}: <b>%{{y:.1f}}%</b><extra></extra>")
            theme.apply_layout(fig, show_legend=True)
            fig.update_layout(title=f"Training by {grain.split(' ')[0]}", height=380)
            fig.update_yaxes(range=[0, 105], ticksuffix="%")
            theme.show(fig)
        with st.container(border=True):
            st.caption(f"Response rate by shop and {grain.split(' ')[0].lower()}.")
            pivot = (summary(grid_rows, ["Location", "Period"])["Response Rate %"]
                     .unstack("Period").reindex(columns=periods))
            pivot["Overall"] = shops.set_index("Location")["Response Rate %"]
            grid.filterable_table(pivot.reset_index(), pinned_columns=("Location",),
                                  total=False, height=420)

    elif view == "✅ Tracker":
        _training_ranking(grid_rows, summary)
        c1, _ = st.columns(2)
        show = c1.selectbox("Show", ["Everyone", "Missed at least one", "Missed the latest"],
                            key="training_show")
        cell = grid_rows.assign(Mark=[
            (f"{s:.0f}" + ("" if t else " (late)")) if a else "✗ missed"
            for a, s, t in zip(grid_rows["Answered"], grid_rows["Score %"].fillna(0),
                               grid_rows["On Time"].fillna(1).astype(bool))])
        matrix = cell.pivot_table(index=["Name", "Location"], columns="Period", values="Mark",
                                  aggfunc=lambda v: " / ".join(v)).reindex(columns=periods)
        per = summary(grid_rows, ["Name", "Location"])[["Answered", "Missed", "Response Rate %"]]
        matrix = matrix.join(per).reset_index()
        if show == "Missed at least one":
            matrix = matrix[matrix["Missed"] > 0]
        elif show == "Missed the latest":
            gone = set(zip(missed_latest["Name"], missed_latest["Location"]))
            matrix = matrix[[(n, l) in gone for n, l in zip(matrix["Name"], matrix["Location"])]]
        st.caption("Every person on the team list against every questionnaire: the score they "
                   "got, \"(late)\" when sent after the day, or ✗ when not answered.")
        with st.container(border=True):
            grid.filterable_table(matrix.sort_values(["Response Rate %", "Location"]),
                                  pinned_columns=("Name",), height=520)
        if not missed_latest.empty:
            st.subheader(f"Not Yet Answered — {latest:%A %d %b}")
            with st.container(border=True):
                grid.filterable_table(
                    missed_latest[["Name", "Location", "Email"]].sort_values(["Location", "Name"]),
                    pinned_columns=("Name",), total=False, height=320)

    else:
        people = summary(grid_rows, ["Name", "Location"])
        lowest = grid_rows[grid_rows["Answered"]].groupby(["Name", "Location"])["Score %"].min()
        last = grid_rows[grid_rows["Answered"]].groupby(["Name", "Location"])["Day"].max()
        people["Lowest Score %"] = lowest
        people["Last Answered"] = last
        people = people.drop(columns=["Team"]).reset_index()
        with st.container(border=True):
            top = people.sort_values(["Response Rate %", "Average Score %"]).tail(30)
            fig = go.Figure()
            fig.add_bar(y=top["Name"] + " — " + top["Location"], x=top["Response Rate %"],
                        orientation="h", name="Response rate",
                        marker=dict(color=theme.CATEGORICAL[0], cornerradius=4))
            fig.add_bar(y=top["Name"] + " — " + top["Location"], x=top["Average Score %"],
                        orientation="h", name="Average score",
                        marker=dict(color=theme.CATEGORICAL[1], cornerradius=4))
            theme.apply_layout(fig, show_legend=True)
            fig.update_layout(title="Respondents — Response Rate and Average Score",
                              barmode="group", hovermode="closest",
                              height=max(420, 34 * len(top) + 100))
            fig.update_xaxes(range=[0, 105], ticksuffix="%")
            theme.show(fig)
        with st.container(border=True):
            st.caption("Each person on the team list over the range: how many questionnaires "
                       "they answered, how fully, how well, and when they last answered.")
            grid.filterable_table(people.sort_values(["Response Rate %", "Average Score %"],
                                                     ascending=False),
                                  pinned_columns=("Name",), height=480)
        strangers = answers[~answers["On List"]] if not answers.empty else answers
        if not strangers.empty:
            st.caption(f"{strangers['Email'].nunique()} people answered but are not on the "
                       "team list (Sheet1): " + ", ".join(sorted(strangers["Email"].unique()))
                       + ". Add them to the list to track them.")


def _training_ranking(grid_rows: pd.DataFrame, summary) -> None:
    """Shops or attendants ranked best first by Training Score: every
    questionnaire due at the score it got, a missed one at 0 — the same
    Training % the score cards use, so answering and scoring well both count."""
    st.subheader("🏆 Training Ranking")
    level = st.segmented_control("Rank", ["🏪 Shops", "🧑‍💼 Attendants"], default="🏪 Shops",
                                 key="training_rank_level",
                                 label_visibility="collapsed") or "🏪 Shops"
    by = ["Location"] if level == "🏪 Shops" else ["Name", "Location"]
    table = summary(grid_rows, by)
    table.insert(0, "Training Score %",
                 grid_rows.groupby(by)["Score %"].apply(lambda v: v.fillna(0).mean()).round(1))
    table = (table.reset_index()
             .sort_values(["Training Score %", "Response Rate %", "Average Score %"],
                          ascending=False, na_position="last"))
    table.insert(0, "Rank", range(1, len(table) + 1))
    if level == "🏪 Shops":
        table = table.rename(columns={"Team": "Team Size"})
    else:
        table = table.drop(columns=["Team"])
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    name_col = "Location" if level == "🏪 Shops" else "Name"
    podium = table.head(3)
    for col, (_, r) in zip(st.columns(3), podium.iterrows()):
        with col.container(border=True):
            label = r[name_col] if level == "🏪 Shops" else f"{r['Name']} — {r['Location']}"
            st.metric(f"{medals[r['Rank']]} {label}", f"{r['Training Score %']:.1f}%",
                      f"{int(r['Answered'])} of {int(r['Answers Due'])} answered · "
                      f"avg score {r['Average Score %']:.0f}%", delta_color="off")
    with st.container(border=True):
        st.caption("Best first. **Training Score** counts every questionnaire due at the score "
                   "it got and a missed one at 0, so it rewards answering and scoring well "
                   "together; ties go to the higher response rate.")
        grid.filterable_table(table, pinned_columns=(name_col,), total=False,
                              height=min(140 + 42 * len(table), 520))


def _kpis_row(items: list[tuple[str, str, str | None]]) -> None:
    for col, (label, value, note) in zip(st.columns(len(items)), items):
        with col.container(border=True):
            st.metric(label, value, note, delta_color="off")


def _card(title: str, row: pd.Series, items: list[tuple[str, str, str | None]],
          measures: list) -> None:
    """One score card: the weighted score, the measures, then each measure's
    score with its weight, and which measures were left out."""
    with st.container(border=True):
        badge = "—" if pd.isna(row["Score"]) else f"{row['Score']:.0f}/100"
        covered = row.get("Weight Covered %")
        note = (f" &nbsp;·&nbsp; scored on {covered:.0f}% of the weighting"
                if pd.notna(covered) and covered < 100 else "")
        label = "Score" if len({m[2] for m in measures}) == 1 else "Weighted score"
        st.markdown(f"**{title}** &nbsp; · &nbsp; {label} **{badge}**{note}")
        for start in range(0, len(items), 4):
            for col, (label, value, sub) in zip(st.columns(4), items[start:start + 4]):
                col.metric(label, value, sub, delta_color="off")
        parts, left_out = [], []
        for column, _, weight in measures:
            name = column.replace(" Score", "")
            if pd.notna(row.get(column)):
                parts.append(f"{name} **{row[column]:.0f}**{_weight_label(measures, weight)}")
            else:
                left_out.append(name)
        equal = len({m[2] for m in measures}) == 1
        st.caption(("Scores out of 100, each counting equally — " if equal
                    else "Scores out of 100 (weight) — ") + " · ".join(parts)
                   + (f". Left out, no data: {', '.join(left_out)}." if left_out else ""))


def _money(v) -> str:
    return "—" if pd.isna(v) else f"KES {v:,.0f}"


def _pct(v) -> str:
    return "—" if pd.isna(v) else f"{v:.1f}%"


def _weight_label(measures: list, weight) -> str:
    """ " (30%)", or nothing when every measure counts the same."""
    return "" if len({m[2] for m in measures}) == 1 else f" ({weight}%)"


_PALETTE = theme.CATEGORICAL + [theme.OTHER, theme.TEXT_SECONDARY, theme.SEQUENTIAL_BLUE[2]]


def _cumulative_chart(table: pd.DataFrame, name: str, title: str, measures: list) -> None:
    """Each bar is the weighted score, split into what each measure adds to it
    (its score × its weight, over the weight the person has data for)."""
    ranked = table.dropna(subset=["Score"]).sort_values("Score").tail(30)
    if ranked.empty:
        return
    covered = ranked["Weight Covered %"].where(ranked["Weight Covered %"] > 0)
    fig = go.Figure()
    for (column, _, weight), color in zip(measures, _PALETTE):
        if column not in ranked or ranked[column].isna().all():
            continue
        part = ranked[column].fillna(0) * weight / covered
        fig.add_bar(y=ranked[name], x=part, orientation="h",
                    name=f"{column.replace(' Score', '')}{_weight_label(measures, weight)}",
                    marker=dict(color=color, line=dict(color=theme.CHART_SURFACE, width=1)),
                    customdata=ranked[[column]],
                    hovertemplate=(f"<b>%{{y}}</b><br>{column}{_weight_label(measures, weight)}: "
                                   "<b>%{customdata[0]:.0f}</b>/100 · adds %{x:.1f}<extra></extra>"))
    for label, score in zip(ranked[name], ranked["Score"]):
        fig.add_annotation(x=score, y=label, text=f"<b>{score:.0f}</b>", showarrow=False,
                           xanchor="left", xshift=6, font=dict(color=theme.TEXT_PRIMARY, size=11))
    theme.apply_layout(fig, show_legend=True)
    fig.update_layout(title=title, barmode="stack", hovermode="closest",
                      legend_traceorder="normal", height=max(420, 28 * len(ranked) + 150))
    fig.update_xaxes(range=[0, 108], showgrid=True, gridcolor=theme.GRIDLINE)
    with st.container(border=True):
        theme.show(fig)
        st.caption("Each bar is the weighted score; the coloured parts show what each measure "
                   "adds to it (its score × its weight). Measures a person had no part in are "
                   "left out and the rest scaled up to 100%.")


def _whatsapp_chats(start_date: date, end_date: date) -> pd.DataFrame | None:
    """Every WhatsApp chat logged in the WhatsApp Monitor in the range."""
    try:
        chats = db.run_query(queries.WA_INTERACTIONS,
                             {"start_date": start_date, "end_date": end_date})
    except Exception:  # the dashboard user cannot read the monitor's tables
        return None
    return chats if not chats.empty else None


def _response_times(chats: pd.DataFrame | None) -> pd.DataFrame | None:
    """Average first response to WhatsApp chats per shop."""
    if chats is None:
        return None
    chats = chats.assign(Minutes=pd.to_numeric(chats["First Response (min)"], errors="coerce"))
    g = chats.dropna(subset=["Minutes"]).groupby("Branch")["Minutes"]
    return pd.DataFrame({"Response Time (min)": g.mean().round(1), "Chats": g.size()})


def _walkins(start_date: date, end_date: date) -> pd.DataFrame | None:
    """Walk-ins per shop and day — the footfall module, with the Reporting
    sheet filling its gaps (the Footfall Conversion page's rules)."""
    params = {"start_date": start_date, "end_date": end_date}
    try:
        system = footfall.prepare_system(db.run_query(queries.FOOTFALL_SYSTEM_DAILY, params))
    except Exception:
        system = footfall.prepare_system(None)
    try:
        foot, rev = footfall.load()
    except footfall.SheetUnavailable:
        foot = pd.DataFrame(columns=["Shop", "Date", "Purchased", "Not Purchased"])
        rev = pd.DataFrame(columns=["Shop", "Date", "Revenue"])
    days = footfall.combine(foot, rev, system, pd.DataFrame(columns=["Shop", "Date", "Target"]),
                            start_date, end_date)
    if days.empty:
        return None
    out = days[days["Walk-ins"] > 0][["Shop", "Date", "Walk-ins"]].rename(columns={"Shop": "Location"})
    out["Location"] = out["Location"].replace({v: k for k, v in footfall.SYSTEM_NAMES.items()})
    return out


def _training_scores(df: pd.DataFrame, start_date: date, end_date: date):
    """Sales training per shop and per attendant (matched to the team list
    by name at their shop) over the closed questionnaires in the range, and
    each shop's completion: the share of its team's questionnaires answered."""
    try:
        data = training.load()
    except training.TrainingUnavailable as exc:
        return None, None, None, f"Sales training is left out: {exc}"
    days = [d for d in data["days"] if start_date <= d <= end_date and d < date.today()]
    if not days:
        return None, None, None, ("No daily training questionnaire closed in this range, so "
                                  "Sales Training is not scored.")
    rows = training.due(data, days)
    score = lambda v: v.fillna(0).mean() * 1.0
    shop = rows.groupby("Location")["Score %"].apply(score)
    completion = rows.groupby("Location")["Answered"].mean() * 100
    by_email = rows.groupby("Email")["Score %"].apply(score)
    pairs = df.loc[df["Served By"] != scorecards.NOT_RECORDED, ["Served By", "Location"]].drop_duplicates()
    emails = [training.match_person(p, loc, data["roster"])
              for p, loc in zip(pairs["Served By"], pairs["Location"])]
    person = pd.Series([by_email.get(e) if e else None for e in emails],
                       index=pd.MultiIndex.from_frame(pairs), dtype=float)
    unmatched = int(pd.isna(person).sum())
    note = (f"Sales Training covers {len(days)} questionnaire(s). {unmatched} attendant(s)/BA(s) "
            "could not be matched to the training team list by name and are scored without it."
            if unmatched else None)
    return shop, person, completion, note


def _match_forms(orders: pd.DataFrame, not_customers: set[str], end_date: date,
                 responses: pd.DataFrame | None) -> pd.DataFrame:
    """Feedback forms matched to sales (scorecards.match_feedback).

    Forms are matched against every order from the start of the range to
    today, not just the range: a form sent after the period still counts for
    its purchase, and one about a later purchase isn't credited to an
    earlier one."""
    if responses is None or responses.empty:
        return pd.DataFrame(columns=["Form", "Order ID", "Submitted", "Staff Rating", "How"])
    later = (db.run_query(queries.SCORECARD_ORDERS,
                          {"start_date": end_date + timedelta(days=1), "end_date": date.today()})
             if end_date < date.today() else orders.iloc[0:0])
    pool = scorecards.prepare(pd.concat([orders, later], ignore_index=True), not_customers)
    return scorecards.match_feedback(pool, responses)


def _optional(sql: str, params: dict | None = None) -> pd.DataFrame | None:
    """A query for one score-card measure: if it fails, the measure is left
    out rather than the whole section."""
    try:
        return db.run_query(sql, params or {})
    except Exception:
        return None


def _scorecard_data(start_date: date, end_date: date) -> dict | None:
    """Shop manager and attendant score cards for the range (None if nothing
    sold), shared by the Score Cards section and the exported report."""
    params = {"start_date": start_date, "end_date": end_date}
    orders = db.run_query(queries.SCORECARD_ORDERS, params)
    if orders.empty:
        return None
    # The same number of days just before, for retention: 8-14 Oct is set
    # against 1-7 Oct, October 1-11 against 20-30 September.
    days = (end_date - start_date).days + 1
    prev_end = start_date - timedelta(days=1)
    prev_start = prev_end - timedelta(days=days - 1)
    prev_orders = db.run_query(queries.SCORECARD_ORDERS,
                               {"start_date": prev_start, "end_date": prev_end})
    not_customers = scorecards.placeholders(orders, prev_orders)
    df = scorecards.prepare(orders, not_customers)
    previous = scorecards.prepare(prev_orders, not_customers)
    try:
        responses = feedback.load()
        feedback_note = None
    except feedback.FeedbackUnavailable as exc:
        responses, feedback_note = None, str(exc)
    matches = _match_forms(orders, not_customers, end_date, responses)
    df = scorecards.apply_feedback(df, matches if responses is not None else None)
    rows = db.run_query(queries.SALES_TARGET_ROWS, params)
    basis = targets.basis_for(start_date, end_date, rows)
    shop_training, person_training, completion, training_note = _training_scores(
        df, start_date, end_date)
    day_targets = targets.daily(rows, start_date, end_date)
    chats = _whatsapp_chats(start_date, end_date)
    walkins = _walkins(start_date, end_date)
    shops = scorecards.shop_cards(
        df, previous, day_targets, shop_training, _response_times(chats),
        training_completion=completion, walkins=walkins, wa=chats,
        stock=_optional(queries.STOCK_COUNTS, params), heads=_optional(queries.SHOP_HEADS))
    people = scorecards.attendant_cards(
        df, previous, day_targets, person_training, walkins=walkins, wa=chats,
        attendance=_optional(queries.ATTENDANCE_DAYS, params))
    matching = _feedback_matching(shops, df, matches, responses, start_date, end_date)
    return {"shops": shops, "shop_cards": scorecards.shop_overall(shops), "people": people,
            "basis": basis, "prev_start": prev_start, "prev_end": prev_end,
            "feedback_note": feedback_note, "training_note": training_note,
            "feedback_matching": matching}


def _feedback_matching(shops: pd.DataFrame, df: pd.DataFrame, matches: pd.DataFrame,
                       responses: pd.DataFrame | None, start_date: date,
                       end_date: date) -> pd.DataFrame | None:
    """Per shop: forms that named it in the period, against the forms matched
    to its orders in the period and how — adds the counts to the shop rows
    and returns the table."""
    if responses is None or responses.empty:
        return None
    day = responses["Submitted"].dt.date
    named = responses[(day >= start_date) & (day <= end_date)]
    received = named["Branch"].astype(str).str.strip().str.lower().value_counts()
    hits = matches.merge(df[["Order ID", "Location"]], on="Order ID")
    how = hits.pivot_table(index="Location", columns="How", values="Form", aggfunc="count").fillna(0)
    table = pd.DataFrame({"Location": shops["Location"]})
    table["Forms Received"] = table["Location"].str.lower().map(received).fillna(0).astype(int)
    for col in ("Phone", "Near-miss phone", "Name"):
        table[f"Matched by {col}"] = table["Location"].map(how.get(col, pd.Series(dtype=float))).fillna(0).astype(int)
    table["Forms Matched"] = table[[c for c in table if c.startswith("Matched by")]].sum(axis=1)
    shops[["Forms Received", "Forms Matched"]] = table[["Forms Received", "Forms Matched"]].to_numpy()
    return table.sort_values("Forms Received", ascending=False)


SHOP_CARD_COLUMNS = (["Location", "Score"] + scorecards.SHOP_SCORE_COLUMNS
                     + ["Revenue", "Target", "Target %", "Corporate Revenue", "Corporate Target",
                        "Corporate %", "Orders", "Basket Value", "Basket Value %", "Bags",
                        "Basket Size", "Attendants", "Revenue per Attendant", "Customers",
                        "Repeat %", "Previous Customers", "Retained", "Retention %", "Feedback",
                        "Feedback %", "Training %", "Chats", "Response Time (min)"])


def _shop_cards_view(data: dict, location: str, due: str, basket: str) -> None:
    """The shops' own score cards: each shop's business as a whole."""
    shops = data["shop_cards"]
    shown = shops if location == "All Locations" else shops[shops["Location"] == location]
    st.caption(
        "**The shop as a whole** — every measure counts equally. **Revenue** against the shop's "
        f"daily targets, **corporate revenue** against {scorecards.CORPORATE_SHARE:.0%} of the "
        f"revenue target, **basket value** against {basket}, **feedback** against "
        f"{scorecards.FEEDBACK_TARGET}% of customers sending a form, **training** (missed "
        "questionnaires at 0) against 100%, **WhatsApp response** against "
        f"{scorecards.RESPONSE_TARGET_MIN} min. Basket size, repeat purchase and retention have "
        "no target, so they are ranked against the other shops: 100 is the best shop. "
        "**Repeat purchase** is customers buying on two or more days in the period; "
        f"**retention** is the previous period's customers ({data['prev_start']:%d %b} – "
        f"{data['prev_end']:%d %b}) buying again at the same shop.")
    _cumulative_chart(shown, "Location", "Shop Score Cards", scorecards.SHOP_MEASURES)
    for _, r in shown.iterrows():
        _card(r["Location"], r, [
            ("Revenue", _money(r["Revenue"]),
             f"{_pct(r['Target %'])} of {_money(r['Target'])} ({due})"
             if pd.notna(r["Target %"]) else "no target"),
            ("Corporate Revenue", _money(r["Corporate Revenue"]),
             f"{_pct(r['Corporate %'])} of {_money(r['Corporate Target'])}"
             if pd.notna(r["Corporate Target"]) else "no target"),
            ("Basket Value", _money(r["Basket Value"]),
             f"{_pct(r['Basket Value %'])} of {basket}"),
            ("Basket Size", f"{r['Basket Size']:.2f} bags", f"{int(r['Orders']):,} orders"),
            ("Revenue per Attendant", _money(r["Revenue per Attendant"]),
             f"{int(r['Attendants'])} attendants/BAs"),
            ("Repeat Purchase", _pct(r["Repeat %"]),
             f"of {r['Customers']:,.0f} customers" if pd.notna(r["Customers"]) else None),
            ("Retention", _pct(r["Retention %"]),
             f"{r['Retained']:,.0f} of {r['Previous Customers']:,.0f} came back"
             if pd.notna(r["Previous Customers"]) else None),
            ("Customer Feedback", _pct(r["Feedback %"]),
             f"{r['Feedback']:,.0f} customers · target {scorecards.FEEDBACK_TARGET}%"
             if pd.notna(r["Feedback"]) else None),
            ("Sales Training", _pct(r["Training %"]), "target 100%"),
            ("WhatsApp Response", "—" if pd.isna(r["Response Time (min)"])
             else f"{r['Response Time (min)']:.1f} min",
             f"average first reply · target {scorecards.RESPONSE_TARGET_MIN} min"),
        ], scorecards.SHOP_MEASURES)
    with st.container(border=True):
        st.caption("All shops side by side, best score first, with each measure's score and the "
                   "figures behind it.")
        grid.filterable_table(shops[[c for c in SHOP_CARD_COLUMNS if c in shops]],
                              pinned_columns=("Location",),
                              currency_columns=("Revenue", "Target", "Corporate Revenue",
                                                "Corporate Target", "Basket Value",
                                                "Revenue per Attendant"),
                              total=False, height=460)


def _weights_text(measures: list) -> str:
    return " · ".join(f"{c.replace(' Score', '')} {w}%" for c, _, w in measures)


def _score_cards(start_date: date, end_date: date, location: str, person: str) -> None:
    data = _scorecard_data(start_date, end_date)
    if data is None:
        st.info("No sales in this date range.")
        return
    feedback_note, training_note = data["feedback_note"], data["training_note"]
    basket = f"KES {scorecards.BASKET_TARGET:,}"
    last = min(end_date, date.today())
    days_so_far = max((last - start_date).days + 1, 0)
    due = f"daily targets, {days_so_far} day{'s' if days_so_far != 1 else ''}"

    views = ["🏬 Shop Score Cards", "👔 Shop Manager Score Cards", "🧑‍💼 Attendant Score Cards"]
    view = st.segmented_control("Score cards", views, default=views[0], key="score_view",
                                label_visibility="collapsed")
    view = view if view in views else views[0]
    st.caption(
        "Every measure is scored out of 100 — the measure against its target, capped at 100 — "
        "and the card's **weighted score** adds them up by their weights. Roles change from day "
        "to day (a desk/printing person at Hilton and KTDA, whoever is on WhatsApp), so each "
        "measure only counts the days a person did that work, read from the orders they were "
        "recorded on: a **floor day** is a day they served a walk-in order, a **WhatsApp day** "
        "a day they served one of the shop's online orders. A measure someone had no part in "
        "over the period is left out and the rest scaled up to 100%, so nobody loses points for "
        "work they were not given. Revenue targets are the shops' daily targets for the "
        f"{days_so_far} day(s) so far ({start_date:%d %b} – {last:%d %b}).")
    if feedback_note:
        st.warning(f"Feedback and customer service are left out: {feedback_note}", icon="🔒")
    if training_note:
        st.caption(training_note)

    if view == "🏬 Shop Score Cards":
        _shop_cards_view(data, location, due, basket)
        return

    if view == "👔 Shop Manager Score Cards":
        shops = data["shops"]
        shown = shops if location == "All Locations" else shops[shops["Location"] == location]
        st.caption(
            f"**Shop manager (the shop's Head in Odoo) — weights:** "
            f"{_weights_text(scorecards.HEAD_MEASURES)}. **Walk-in conversion** is walk-in "
            "orders over walk-ins counted (footfall module, the Reporting sheet filling gaps) "
            "and **WhatsApp conversion** chats that bought over chats — both against the "
            "network's rate (at or above it scores 100). **Response time** is the average "
            f"first reply against {scorecards.RESPONSE_TARGET_MIN} min. **Feedback** is "
            f"customers sending a form against {scorecards.FEEDBACK_TARGET}%. **Training "
            "completion** is the share of the team's questionnaires answered. **Protect "
            "stock** is the variance the latest stock counts found (counts submitted in the "
            f"period or the 2 weeks before): 0% scores 100, {scorecards.STOCK_VARIANCE_ZERO_AT:.0f}% "
            "or more off scores 0.")
        _cumulative_chart(shown.assign(Name=shown["Location"] + " — " + shown["Head"].fillna("no head set")),
                          "Name", "Shop Manager Score Cards — Weighted", scorecards.HEAD_MEASURES)
        for _, r in shown.iterrows():
            head = r["Head"] if pd.notna(r["Head"]) else "no head set in Odoo"
            _card(f"{r['Location']} — {head}", r, [
                ("Shop Target", _money(r["Revenue"]),
                 f"{_pct(r['Target %'])} of {_money(r['Target'])} ({due})"
                 if pd.notna(r["Target %"]) else "no target"),
                ("Basket Value", _money(r["Basket Value"]),
                 f"{_pct(r['Basket Value %'])} of {basket}"),
                ("Walk-in Conversion", _pct(r["Walk-in Conversion %"]),
                 f"{r['Walk-in Orders']:,.0f} of {r['Walk-ins']:,.0f} walk-ins"
                 if pd.notna(r["Walk-ins"]) else "no footfall counted"),
                ("WhatsApp Conversion", _pct(r["WhatsApp Conversion %"]),
                 f"{r['WhatsApp Converted']:,.0f} of {r['WhatsApp Chats']:,.0f} chats"
                 if pd.notna(r["WhatsApp Chats"]) else "no chats logged"),
                ("Response Time", "—" if pd.isna(r["Response Time (min)"])
                 else f"{r['Response Time (min)']:.1f} min",
                 f"target {scorecards.RESPONSE_TARGET_MIN} min"),
                ("Customer Feedback", _pct(r["Feedback %"]),
                 f"{r['Feedback']:,.0f} customers · {r.get('Forms Received', 0):,.0f} forms named the shop"
                 if pd.notna(r["Feedback"]) else None),
                ("Training Completion", _pct(r["Training Completion %"]),
                 f"average score {_pct(r['Training %'])}"),
                ("Protect Stock", "—" if pd.isna(r["Stock Variance %"])
                 else f"{r['Stock Variance %']:.1f}% variance",
                 f"last count {pd.Timestamp(r['Last Stock Count']):%d %b}"
                 if pd.notna(r["Last Stock Count"]) else "no stock count"),
            ], scorecards.HEAD_MEASURES)
        matching = data.get("feedback_matching")
        if matching is not None:
            with st.expander("Customer feedback: forms received vs matched to a sale"):
                st.caption(
                    "Forms that named each shop in the period, and the forms matched to one of its "
                    "sales in the period. A form is matched to the purchase it is about — the "
                    f"customer's latest order before it, up to {scorecards.FEEDBACK_WINDOW_DAYS} "
                    "days, so forms sent after the period still count. When the phone on the "
                    "form isn't the one on the POS, it is matched by **name** (the first name on "
                    "the form against customer names on the shop's orders in the "
                    f"{scorecards.NAME_MATCH_DAYS} days before), and a number one or two "
                    "digits off a customer's at the shop counts as theirs. Forms received can "
                    "still be higher: forms about a purchase before the period, and forms with "
                    "neither a matching number nor name.")
                grid.filterable_table(matching, pinned_columns=("Location",), height=360)
        with st.container(border=True):
            st.caption("All shops side by side, best weighted score first, with each measure's "
                       "score and the figures behind it.")
            grid.filterable_table(shops[SCORE_SHOP_COLUMNS], pinned_columns=("Location",),
                                  currency_columns=("Revenue", "Target", "Corporate Revenue",
                                                    "Basket Value", "Revenue per Attendant"),
                                  total=False, height=460)
        return

    people = data["people"]
    if location != "All Locations":
        people = people[people["Location"] == location]
    st.caption(
        f"**Shop attendant / BA — weights:** {_weights_text(scorecards.ATTENDANT_MEASURES)}. "
        "**Personal target**: each floor day the shop's daily target is shared equally between "
        "everyone serving that day (a KES 100,000 day with two people gives each 50,000), "
        "summed over their days; the % is their own sales over it. **Bulk orders** are "
        f"corporate orders and orders of {scorecards.BULK_MIN_BAGS}+ bags, against "
        f"{scorecards.BULK_SHARE:.0%} of their target. **Conversion** is their walk-in orders "
        "over the customers they served — the shop's walk-ins each floor day split between "
        "everyone serving walk-ins that day — against the network's rate. **New products** is "
        "the share of their sales from bags Odoo flags as new, against the network's share. "
        "**Respond fast** is the shop's WhatsApp chats answered within "
        f"{scorecards.RESPONSE_TARGET_MIN} min on their WhatsApp days. **Customer service** is "
        "the staff rating (1-5) on the feedback forms of customers they served. **Capture** is "
        "their orders with a real customer phone number. **Be there** is being on time on the "
        f"days they worked (late counts {scorecards.LATE_CREDIT:.0%}), for people who clock in "
        "— most shop staff don't yet, so it is left out for them.")
    if people.empty:
        st.info("No attendants or BAs recorded in this range.")
        return
    people = people.assign(Name=people["Attendant / BA"] + " — " + people["Location"])
    chosen = people if person == ALL_PEOPLE else people[people["Attendant / BA"] == person]
    _cumulative_chart(chosen, "Name", "Attendant Score Cards — Weighted",
                      scorecards.ATTENDANT_MEASURES)
    for _, r in chosen.head(12).iterrows():
        _card(r["Name"], r, [
            ("Personal Target", _money(r["Revenue"]),
             f"{_pct(r['Target %'])} of {_money(r['Target'])} · {int(r['Days Worked'])} days"
             if pd.notna(r["Target %"]) else "no target"),
            ("Bulk Orders", _money(r["Bulk Revenue"]),
             f"{_pct(r['Bulk %'])} of {_money(r['Bulk Target'])}"
             if pd.notna(r["Bulk Target"]) else None),
            ("Basket Value", _money(r["Basket Value"]),
             f"{_pct(r['Basket Value %'])} of {basket}"),
            ("New Products", _pct(r["New Products %"]), "of their sales"),
            ("Conversion", _pct(r["Conversion %"]),
             f"{r['Walk-in Orders']:,.0f} of {r['Customers Served']:,.0f} served"
             if pd.notna(r["Customers Served"]) else "no floor days with footfall"),
            ("Respond Fast", _pct(r["Fast Reply %"]),
             f"{r['WhatsApp Days']:,.0f} WhatsApp day(s)" if pd.notna(r["WhatsApp Days"])
             else "not on WhatsApp"),
            ("Customer Service", "—" if pd.isna(r["Service Rating"])
             else f"{r['Service Rating']:.2f} / 5",
             f"{r['Ratings']:,.0f} rating(s)" if pd.notna(r["Ratings"]) else "no ratings"),
            ("Customer Feedback", _pct(r["Feedback %"]),
             f"{r['Feedback']:,.0f} customers · target {scorecards.FEEDBACK_TARGET}%"
             if pd.notna(r["Feedback"]) else None),
            ("Capture the Customer", _pct(r["Capture %"]),
             f"{r['Captured Orders']:,.0f} of {int(r['Orders']):,} orders"),
            ("Know the Product", _pct(r["Training %"]),
             "training score" if pd.notna(r["Training %"]) else "not matched to the training list"),
            ("Be There", _pct(r["Attendance %"]),
             f"{r['Late Days']:,.0f} late day(s)" if pd.notna(r["Attendance %"])
             else "no clock-ins"),
        ], scorecards.ATTENDANT_MEASURES)
    if person == ALL_PEOPLE and len(people) > 12:
        st.caption(f"The 12 best weighted scores are shown as cards; all {len(people)} are in "
                   "the table. Pick a person at the top to see their card.")
    with st.container(border=True):
        grid.filterable_table(people[SCORE_PEOPLE_COLUMNS], pinned_columns=("Attendant / BA",),
                              currency_columns=("Target", "Revenue", "Revenue per Day",
                                                "Bulk Revenue", "Bulk Target", "New Bag Revenue",
                                                "Corporate Revenue", "Basket Value"),
                              total=False, height=460)


render_attendants(start_date, end_date)
