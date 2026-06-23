
from pathlib import Path
import json
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Narrative Intelligence Engine",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

def find_file(name):
    for path in [DATA_DIR / name, APP_DIR / name]:
        if path.exists():
            return path
    return None

@st.cache_data(show_spinner=False)
def load_csv(name):
    path = find_file(name)
    if path is None:
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_json(name):
    path = find_file(name)
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

articles = load_csv("article_scores_final.csv")
clusters = load_csv("narrative_cluster_summary_final.csv")
agenda = load_csv("agenda_shapers_showcase.csv")
outlets = load_csv("outlet_scores_final.csv")
evidence = load_csv("evidence_snippets_showcase.csv")
positions = load_csv("debate_position_table_showcase.csv")
overview = load_json("overview_showcase.json")

cluster_ts = load_csv("cluster_timeseries_final.csv")
stance_ts = load_csv("stance_timeseries_final.csv")
frame_ts = load_csv("frame_timeseries_final.csv")
overall_ts = load_csv("overall_stance_timeseries_final.csv")
daily_cluster_ts = load_csv("daily_cluster_timeseries_final.csv")
daily_overall_ts = load_csv("daily_overall_timeseries_final.csv")
agenda_ts = load_csv("agenda_timeseries_final.csv")
readiness = load_csv("intervention_readiness_final.csv")

opinion_articles = load_csv("opinion_attribution_articles_v2.csv")
contributor_attribution = load_csv("contributor_attribution_summary_v2.csv")
opinion_outlet_balance = load_csv("outlet_attribution_balance_v2.csv")
opinion_overview = load_json("opinion_attribution_overview_v2.json")

st.markdown("""
<style>
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
.hero {
  padding: 1.5rem 1.6rem; border-radius: 28px;
  background: linear-gradient(135deg,#062f2f 0%,#0f766e 55%,#facc15 160%);
  color: white; margin-bottom: 1rem; box-shadow: 0 22px 55px rgba(15,118,110,.25);
}
.hero h1 { color:white; margin:0 0 .35rem 0; font-size:2.4rem; letter-spacing:-.04em; }
.hero p { color:#ecfffb; font-size:1.05rem; max-width:1050px; }
.card {
  background:white; border:1px solid #d8ece9; border-radius:24px; padding:1.05rem 1.1rem;
  box-shadow:0 12px 34px rgba(15,118,110,.08); margin-bottom:1rem;
}
.metric {
  background:white; border:1px solid #d8ece9; border-radius:22px; padding:1rem;
  box-shadow:0 12px 34px rgba(15,118,110,.08); min-height:110px;
}
.metric .label { font-size:.75rem; color:#637b78; text-transform:uppercase; letter-spacing:.08em; font-weight:800; }
.metric .value { font-size:1.7rem; color:#063c3a; font-weight:900; line-height:1.1; margin-top:.25rem; }
.metric .note { font-size:.84rem; color:#6b827f; margin-top:.25rem; }
.tag {
  display:inline-block; border-radius:999px; padding:.32rem .62rem;
  background:#e7f7f4; color:#08756f; font-size:.76rem; font-weight:800; margin:.15rem;
}
.tag-dark {
  display:inline-block; border-radius:999px; padding:.32rem .62rem;
  background:#083d3b; color:white; font-size:.76rem; font-weight:800; margin:.15rem;
}
.evidence {
  background:white; border:1px solid #d8ece9; border-left:6px solid #0f766e;
  border-radius:18px; padding:1rem; margin-bottom:.85rem;
}
.formula {
  background:#f4fbfa; border:1px solid #d7efeb; border-radius:18px;
  padding:1rem; color:#08413e; font-family:monospace; white-space:pre-wrap;
}
.note-box {
  background:#fff8e8; border:1px solid #ffd98a; color:#604200;
  border-radius:18px; padding:.9rem 1rem; margin:.65rem 0 1rem 0;
}
.ok-box {
  background:#ecfdf5; border:1px solid #a7f3d0; color:#064e3b;
  border-radius:18px; padding:.9rem 1rem; margin:.65rem 0 1rem 0;
}
</style>
""", unsafe_allow_html=True)

def fmt_num(x):
    try:
        x = float(x)
    except Exception:
        return "0"
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x/1_000:.1f}K"
    if x == int(x):
        return f"{int(x):,}"
    return f"{x:,.1f}"

def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          <div class="note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def stance_bucket(score):
    try:
        score = float(score)
    except Exception:
        return "Mixed / unclear"
    if score <= -0.45:
        return "Strong preserve"
    if score <= -0.15:
        return "Lean preserve"
    if score < 0.15:
        return "Mixed / unclear"
    if score < 0.45:
        return "Reform / redefine"
    return "Move away / align"

def display_table(df, cols, n=None):
    if df.empty:
        st.info("No data available.")
        return
    use_cols = [c for c in cols if c in df.columns]
    out = df[use_cols].copy()
    if n is not None:
        out = out.head(n)
    st.dataframe(out, use_container_width=True, hide_index=True)

def get_cluster_col(df):
    if "cluster_display_name" in df.columns:
        return "cluster_display_name"
    if "cluster_name" in df.columns:
        return "cluster_name"
    return None

def source_link(url):
    if isinstance(url, str) and url.strip() and url.lower() != "nan":
        return f"<a href='{url}' target='_blank'>Open source</a>"
    return ""

def parse_week_start(value):
    if pd.isna(value):
        return pd.NaT
    value = str(value)
    if "/" in value:
        value = value.split("/")[0]
    return pd.to_datetime(value, errors="coerce")

def prep_week(df):
    out = df.copy()
    if not out.empty and "publication_week" in out.columns:
        out["week_start"] = out["publication_week"].apply(parse_week_start)
        out = out[out["week_start"].notna()].copy()
    return out

def numeric(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    return out

st.sidebar.markdown("## 🧭 Narrative Engine")
st.sidebar.markdown("CSV-powered demo dashboard")

pages = [
    "Executive Overview",
    "Narrative Map",
    "Narrative Timeline",
    "Contributor Attribution",
    "Agenda Shapers",
    "Debate Positioning",
    "Outlet Influence",
    "Evidence Explorer",
    "Impact Lab",
    "Methodology",
]
page = st.sidebar.radio("View", pages)
st.sidebar.markdown("---")

cluster_col_articles = get_cluster_col(articles)
cluster_filter = "All"
if not articles.empty and cluster_col_articles:
    cluster_options = ["All"] + sorted(articles[cluster_col_articles].dropna().astype(str).unique().tolist())
    cluster_filter = st.sidebar.selectbox("Narrative cluster", cluster_options)

stance_filter = "All"
if not articles.empty and "stance_bucket" in articles.columns:
    stance_options = ["All"] + sorted(articles["stance_bucket"].dropna().astype(str).unique().tolist())
    stance_filter = st.sidebar.selectbox("Stance", stance_options)

outlet_filter = "All"
if not articles.empty and "Outlet" in articles.columns:
    outlet_options = ["All"] + sorted(articles["Outlet"].dropna().astype(str).unique().tolist())
    outlet_filter = st.sidebar.selectbox("Outlet", outlet_options)

filtered_articles = articles.copy()
if cluster_filter != "All" and cluster_col_articles:
    filtered_articles = filtered_articles[filtered_articles[cluster_col_articles] == cluster_filter]
if stance_filter != "All" and "stance_bucket" in filtered_articles.columns:
    filtered_articles = filtered_articles[filtered_articles["stance_bucket"] == stance_filter]
if outlet_filter != "All" and "Outlet" in filtered_articles.columns:
    filtered_articles = filtered_articles[filtered_articles["Outlet"] == outlet_filter]

st.markdown("""
<div class="hero">
  <h1>Narrative Intelligence Engine</h1>
  <p>Clusters media coverage, weights outlet influence, maps debate stance, ranks agenda shapers, tracks narrative movement over time, and drills into evidence.</p>
</div>
""", unsafe_allow_html=True)

if articles.empty and clusters.empty and agenda.empty:
    st.error("No data files were found. Put the CSV/JSON files either in the repo root or in a data/ folder.")
    st.stop()

if page == "Executive Overview":
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Mentions analysed", fmt_num(len(articles)), "modelled coverage items")
    with c2:
        metric_card("Narrative clusters", fmt_num(clusters["cluster_display_name"].nunique() if "cluster_display_name" in clusters.columns else len(clusters)), "semantic subtopics")
    with c3:
        metric_card("Agenda shapers", fmt_num(len(agenda)), "actors + outlets")
    with c4:
        metric_card("Media outlets", fmt_num(outlets["Outlet"].nunique() if "Outlet" in outlets.columns else 0), "source universe")
    with c5:
        metric_card("Timeline weeks", fmt_num(overview.get("weeks_available", "")) if overview.get("weeks_available", "") != "" else "—", "publication coverage")

    left, right = st.columns([1.4, 1])
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Narrative cluster landscape")
        if not clusters.empty:
            plot = clusters.copy()
            if "avg_stance_score" not in plot.columns:
                plot["avg_stance_score"] = 0
            if "weighted_mentions" not in plot.columns:
                plot["weighted_mentions"] = 1
            if "mention_count" not in plot.columns:
                plot["mention_count"] = 1
            plot["stance_direction"] = plot["avg_stance_score"].apply(stance_bucket)
            fig = px.scatter(
                plot,
                x="mention_count",
                y="weighted_mentions",
                size="weighted_mentions",
                color="stance_direction",
                hover_name="cluster_display_name",
                hover_data=[c for c in ["mention_count", "weighted_mentions", "estimated_reach", "avg_stance_score", "dominant_stance", "dominant_frame"] if c in plot.columns],
                template="plotly_white",
                size_max=60,
                labels={"mention_count": "Mention volume", "weighted_mentions": "Influence-weighted volume"},
            )
            fig.update_layout(height=520, margin=dict(l=10, r=10, t=25, b=10))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Top agenda shapers")
        display_table(
            agenda.sort_values("impact_score", ascending=False) if "impact_score" in agenda.columns else agenda,
            ["entity", "entity_group", "actor_type", "narrative_role", "stance_label", "impact_score"],
            12,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Stance distribution")
        if not filtered_articles.empty and "stance_bucket" in filtered_articles.columns:
            stance_df = filtered_articles["stance_bucket"].value_counts().reset_index()
            stance_df.columns = ["stance", "mentions"]
            fig = px.bar(stance_df, x="mentions", y="stance", orientation="h", template="plotly_white")
            fig.update_layout(height=320, yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=25, b=10))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("What this demo proves")
    st.write(
        "The tool moves beyond basic sentiment. It separates sentiment, stance, frame, influence, actor role, time movement and evidence. Coverage can look neutral in tone while still shifting a debate through repeated framing, actor prominence and outlet influence."
    )
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Narrative Map":
    st.header("Narrative Map")
    st.write("Explore the main subtopics and coverage clusters inside the debate.")
    if clusters.empty:
        st.warning("No cluster data found.")
    else:
        sort_col = "weighted_mentions" if "weighted_mentions" in clusters.columns else "mention_count"
        clusters_sorted = clusters.sort_values(sort_col, ascending=False)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        fig = px.bar(
            clusters_sorted,
            x=sort_col,
            y="cluster_display_name",
            color="avg_stance_score" if "avg_stance_score" in clusters_sorted.columns else None,
            orientation="h",
            hover_data=[c for c in ["mention_count", "estimated_reach", "dominant_stance", "dominant_frame"] if c in clusters_sorted.columns],
            template="plotly_white",
            labels={sort_col: "Weighted coverage", "cluster_display_name": "", "avg_stance_score": "Avg stance"},
        )
        fig.update_layout(height=620, yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=25, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        selected = st.selectbox("Open a narrative cluster", clusters_sorted["cluster_display_name"].tolist())
        sub = articles[articles[cluster_col_articles] == selected] if cluster_col_articles else pd.DataFrame()

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader(selected)
        a, b, c, d = st.columns(4)
        with a:
            metric_card("Mentions", fmt_num(len(sub)))
        with b:
            metric_card("Weighted volume", fmt_num(sub["mention_weight"].sum() if "mention_weight" in sub.columns else 0))
        with c:
            metric_card("Estimated reach", fmt_num(sub["Estimated Reach"].sum() if "Estimated Reach" in sub.columns else 0))
        with d:
            avg = sub["stance_score"].mean() if "stance_score" in sub.columns and len(sub) else 0
            metric_card("Avg stance", f"{avg:.2f}", stance_bucket(avg))
        st.subheader("Top articles in this narrative")
        if not sub.empty:
            rank = "mention_weight" if "mention_weight" in sub.columns else None
            if rank:
                sub = sub.sort_values(rank, ascending=False)
            display_table(sub, ["publication_datetime", "Headline", "Outlet", "Sentiment", "Estimated Reach", "stance_bucket", "dominant_frame", "mention_weight_100", "Link"], 30)
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Narrative Timeline":
    st.header("Narrative Timeline")
    st.write("Shows how coverage volume, frames, stance and agenda activity move over time.")

    if cluster_ts.empty and overall_ts.empty:
        st.warning("No timeline files found. Upload the time-series CSVs generated from the updated dataset.")
    else:
        ready = readiness.iloc[0].to_dict() if not readiness.empty else {}
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Dated articles", fmt_num(ready.get("matched_articles", len(articles))), "used for timeline")
        with c2:
            metric_card("Match rate", f"{float(ready.get('match_rate_pct', 0)):.1f}%", "date coverage")
        with c3:
            metric_card("Weeks", fmt_num(ready.get("weeks_available", overview.get("weeks_available", 0))), "publication weeks")
        with c4:
            metric_card("Days", fmt_num(ready.get("days_available", overview.get("days_available", 0))), "publication days")

        st.markdown("<div class='ok-box'><b>Time upgrade active:</b> timeline uses publication dates from the full modelled dataset.</div>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["Narrative volume", "Stance drift", "Frame movement", "Agenda activity"])

        with tab1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Narrative volume over time")
            weekly = prep_week(cluster_ts)
            if not weekly.empty:
                weekly = numeric(weekly, ["mentions", "weighted_mentions", "estimated_reach", "avg_stance", "avg_sentiment"])
                top_n = st.slider("Number of narratives to show", 3, 12, 7)
                metric_choice = st.radio("Timeline metric", ["weighted_mentions", "mentions", "estimated_reach"], horizontal=True)
                top_clusters = weekly.groupby("cluster_name")[metric_choice].sum().sort_values(ascending=False).head(top_n).index.tolist()
                plot = weekly[weekly["cluster_name"].isin(top_clusters)]
                fig = px.area(
                    plot,
                    x="week_start",
                    y=metric_choice,
                    color="cluster_name",
                    template="plotly_white",
                    labels={"week_start": "Publication week", metric_choice: metric_choice.replace("_", " ").title(), "cluster_name": "Narrative"},
                )
                fig.update_layout(height=580, margin=dict(l=10, r=10, t=25, b=10))
                st.plotly_chart(fig, use_container_width=True)
                display_table(plot.sort_values(["week_start", metric_choice], ascending=[False, False]), ["publication_week", "cluster_name", "mentions", "weighted_mentions", "estimated_reach", "avg_stance", "avg_sentiment"], 100)
            st.markdown("</div>", unsafe_allow_html=True)

        with tab2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Stance drift over time")
            overall = prep_week(overall_ts)
            if not overall.empty:
                overall = numeric(overall, ["mentions", "weighted_mentions", "estimated_reach", "avg_stance", "avg_sentiment"])
                fig = px.line(
                    overall,
                    x="week_start",
                    y="avg_stance",
                    markers=True,
                    template="plotly_white",
                    hover_data=[c for c in ["mentions", "weighted_mentions", "estimated_reach", "avg_sentiment"] if c in overall.columns],
                    labels={"week_start": "Publication week", "avg_stance": "Average stance"},
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(height=430, margin=dict(l=10, r=10, t=25, b=10))
                st.plotly_chart(fig, use_container_width=True)
            buckets = prep_week(stance_ts)
            if not buckets.empty:
                buckets = numeric(buckets, ["mentions", "weighted_mentions", "estimated_reach"])
                fig2 = px.bar(
                    buckets,
                    x="week_start",
                    y="weighted_mentions",
                    color="stance_bucket",
                    template="plotly_white",
                    labels={"week_start": "Publication week", "weighted_mentions": "Weighted mentions", "stance_bucket": "Stance"},
                )
                fig2.update_layout(height=430, margin=dict(l=10, r=10, t=25, b=10))
                st.plotly_chart(fig2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with tab3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Frame movement over time")
            frames = prep_week(frame_ts)
            if not frames.empty:
                frames = numeric(frames, ["mentions", "weighted_mentions", "estimated_reach", "avg_stance"])
                top_n = st.slider("Frames to show", 3, 12, 8, key="frame_slider")
                top_frames = frames.groupby("frame")["weighted_mentions"].sum().sort_values(ascending=False).head(top_n).index.tolist()
                plot = frames[frames["frame"].isin(top_frames)]
                fig = px.line(
                    plot,
                    x="week_start",
                    y="weighted_mentions",
                    color="frame",
                    markers=True,
                    template="plotly_white",
                    labels={"week_start": "Publication week", "weighted_mentions": "Weighted mentions", "frame": "Frame"},
                )
                fig.update_layout(height=560, margin=dict(l=10, r=10, t=25, b=10))
                st.plotly_chart(fig, use_container_width=True)
                display_table(plot.sort_values(["week_start", "weighted_mentions"], ascending=[False, False]), ["publication_week", "frame", "mentions", "weighted_mentions", "estimated_reach", "avg_stance"], 100)
            st.markdown("</div>", unsafe_allow_html=True)

        with tab4:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Agenda activity over time")
            ents = prep_week(agenda_ts)
            if not ents.empty:
                ents = numeric(ents, ["mentions", "unique_stories", "weighted_mentions", "estimated_reach", "avg_stance"])
                top_n = st.slider("Entities to show", 5, 20, 10, key="entity_slider")
                metric_choice = st.radio("Entity metric", ["weighted_mentions", "mentions", "estimated_reach"], horizontal=True, key="entity_metric")
                top_entities = ents.groupby("entity")[metric_choice].sum().sort_values(ascending=False).head(top_n).index.tolist()
                plot = ents[ents["entity"].isin(top_entities)]
                fig = px.line(
                    plot,
                    x="week_start",
                    y=metric_choice,
                    color="entity",
                    markers=True,
                    template="plotly_white",
                    labels={"week_start": "Publication week", metric_choice: metric_choice.replace("_", " ").title(), "entity": "Entity"},
                )
                fig.update_layout(height=580, margin=dict(l=10, r=10, t=25, b=10))
                st.plotly_chart(fig, use_container_width=True)
                display_table(plot.sort_values(["week_start", metric_choice], ascending=[False, False]), ["publication_week", "entity", "mentions", "unique_stories", "weighted_mentions", "estimated_reach", "avg_stance"], 120)
            st.markdown("</div>", unsafe_allow_html=True)

elif page == "Contributor Attribution":
    st.header("Contributor Attribution")
    st.write("Separates publication-owned opinion from reported actor opinion, and shows how real or synthetic contributors carry the debate.")

    if opinion_articles.empty and contributor_attribution.empty and opinion_outlet_balance.empty:
        st.warning("No contributor attribution files found. Upload the V2 opinion attribution CSV/JSON files into the repo root or data/ folder.")
    else:
        total_attr = int(opinion_overview.get("total_articles", len(opinion_articles))) if opinion_overview else len(opinion_articles)

        if opinion_overview:
            publication_owned_pct = float(opinion_overview.get("publication_owned_opinion_pct", 0))
            actor_mediated_pct = float(opinion_overview.get("actor_mediated_pct", 0))
            contested_pct = float(opinion_overview.get("contested_or_multi_actor_pct", 0))
            unclear_pct = float(opinion_overview.get("unclear_attribution_pct", 0))
        else:
            publication_owned_pct = 0
            actor_mediated_pct = 0
            contested_pct = 0
            unclear_pct = 0
            if not opinion_articles.empty and "opinion_attribution_v2" in opinion_articles.columns:
                total_attr = len(opinion_articles)
                publication_owned_pct = opinion_articles["opinion_attribution_v2"].eq("publication_owned_opinion").mean() * 100
                actor_mediated_pct = opinion_articles["opinion_attribution_v2"].isin([
                    "single_actor_reported_opinion",
                    "multi_actor_reported_opinion",
                    "multi_actor_contested_reported_opinion",
                ]).mean() * 100
                contested_pct = opinion_articles["opinion_attribution_v2"].isin([
                    "multi_actor_reported_opinion",
                    "multi_actor_contested_reported_opinion",
                ]).mean() * 100
                unclear_pct = opinion_articles["opinion_attribution_v2"].eq("unclear_attribution").mean() * 100

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            metric_card("Articles attributed", fmt_num(total_attr), "with opinion ownership classification")
        with c2:
            metric_card("Publication-owned", f"{publication_owned_pct:.1f}%", "article/contributor appears to own the stance")
        with c3:
            metric_card("Actor-mediated", f"{actor_mediated_pct:.1f}%", "stance belongs to reported actors")
        with c4:
            metric_card("Multi-actor", f"{contested_pct:.1f}%", "coverage carries more than one side or actor")
        with c5:
            metric_card("Unclear", f"{unclear_pct:.1f}%", "not enough attribution evidence")

        st.markdown(
            "<div class='ok-box'><b>Attribution layer active:</b> this separates the contributor/byline from the true owner of the opinion. Synthetic contributor names are demo placeholders where byline metadata is missing.</div>",
            unsafe_allow_html=True,
        )

        tab1, tab2, tab3, tab4 = st.tabs(["Attribution split", "Contributor profiles", "Outlet balance", "Article evidence"])

        with tab1:
            left, right = st.columns([1.1, 1])
            with left:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Opinion ownership split")
                if not opinion_articles.empty and "opinion_attribution_label" in opinion_articles.columns:
                    attr_counts = opinion_articles["opinion_attribution_label"].fillna("Unclear attribution").value_counts().reset_index()
                    attr_counts.columns = ["Attribution", "Articles"]
                    fig = px.bar(
                        attr_counts,
                        x="Articles",
                        y="Attribution",
                        orientation="h",
                        template="plotly_white",
                        labels={"Articles": "Articles", "Attribution": ""},
                    )
                    fig.update_layout(height=430, yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=25, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                    display_table(attr_counts, ["Attribution", "Articles"])
                else:
                    st.info("Article-level attribution labels not available.")
                st.markdown("</div>", unsafe_allow_html=True)

            with right:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("How to read this")
                st.write(
                    "Publication-owned means the article appears to carry the stance itself. "
                    "Actor-mediated means the article is reporting, quoting, contrasting or amplifying an external actor's stance. "
                    "Multi-actor means the article carries more than one reported actor, side or claim."
                )
                st.write(
                    "This prevents the app from wrongly treating a reported quote as the outlet's own opinion."
                )
                st.markdown("</div>", unsafe_allow_html=True)

        with tab2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Contributor profiles")

            view = contributor_attribution.copy()
            if not view.empty:
                contributor_type_filter = "All"
                if "contributor_type" in view.columns:
                    type_options = ["All"] + sorted(view["contributor_type"].dropna().astype(str).unique().tolist())
                    contributor_type_filter = st.selectbox("Contributor type", type_options)
                    if contributor_type_filter != "All":
                        view = view[view["contributor_type"].astype(str) == contributor_type_filter]

                min_articles = st.slider("Minimum articles", 1, 30, 2)
                if "articles" in view.columns:
                    view = view[pd.to_numeric(view["articles"], errors="coerce").fillna(0) >= min_articles]

                plot = view.copy()
                for c in ["actor_mediated_pct", "contested_or_multi_actor_pct", "publication_owned_pct", "articles"]:
                    if c in plot.columns:
                        plot[c] = pd.to_numeric(plot[c], errors="coerce").fillna(0)

                if not plot.empty and {"actor_mediated_pct", "contested_or_multi_actor_pct", "articles"}.issubset(plot.columns):
                    fig = px.scatter(
                        plot.head(120),
                        x="actor_mediated_pct",
                        y="contested_or_multi_actor_pct",
                        size="articles",
                        color="contributor_type" if "contributor_type" in plot.columns else None,
                        hover_name="contributor_display_name",
                        hover_data=[c for c in ["Outlet", "articles", "publication_owned_pct", "dominant_attribution", "one_sided_amplification_risk", "top_reported_actors"] if c in plot.columns],
                        template="plotly_white",
                        labels={
                            "actor_mediated_pct": "Actor-mediated coverage %",
                            "contested_or_multi_actor_pct": "Multi-actor / contested coverage %",
                            "articles": "Articles",
                        },
                        size_max=45,
                    )
                    fig.update_layout(height=560, margin=dict(l=10, r=10, t=25, b=10))
                    st.plotly_chart(fig, use_container_width=True)

                display_table(
                    view,
                    [
                        "contributor_display_name", "Outlet", "contributor_type", "synthetic_author_flag", "articles",
                        "publication_owned_pct", "actor_mediated_pct", "contested_or_multi_actor_pct",
                        "dominant_attribution", "dominant_stance_side", "viewpoint_spread_score",
                        "one_sided_amplification_risk", "top_reported_actors", "top_frames", "top_clusters",
                    ],
                    100,
                )
            else:
                st.info("No contributor summary found.")
            st.markdown("</div>", unsafe_allow_html=True)

        with tab3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Outlet attribution balance")
            view = opinion_outlet_balance.copy()
            if not view.empty:
                for c in ["actor_mediated_pct", "publication_owned_pct", "contested_or_multi_actor_pct", "total_articles"]:
                    if c in view.columns:
                        view[c] = pd.to_numeric(view[c], errors="coerce").fillna(0)
                sort_col = "actor_mediated_pct" if "actor_mediated_pct" in view.columns else "total_articles"
                view = view.sort_values(sort_col, ascending=False)

                fig = px.bar(
                    view.head(30),
                    x=sort_col,
                    y="Outlet",
                    orientation="h",
                    hover_data=[c for c in ["total_articles", "publication_owned_pct", "contested_or_multi_actor_pct", "one_sided_amplification_risk", "top_reported_actors"] if c in view.columns],
                    template="plotly_white",
                    labels={sort_col: sort_col.replace("_", " ").title(), "Outlet": ""},
                )
                fig.update_layout(height=720, yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=25, b=10))
                st.plotly_chart(fig, use_container_width=True)

                display_table(
                    view,
                    [
                        "Outlet", "total_articles", "publication_owned_pct", "actor_mediated_pct",
                        "contested_or_multi_actor_pct", "factual_low_opinion_pct", "unclear_pct",
                        "preserve_side_pct", "reform_side_pct", "move_away_side_pct", "mixed_unclear_side_pct",
                        "viewpoint_spread_score", "one_sided_amplification_risk", "top_reported_actors", "top_frames",
                    ],
                    100,
                )
            else:
                st.info("No outlet attribution balance found.")
            st.markdown("</div>", unsafe_allow_html=True)

        with tab4:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Article-level attribution evidence")
            view = opinion_articles.copy()
            if not view.empty:
                if "opinion_attribution_label" in view.columns:
                    attr_options = ["All"] + sorted(view["opinion_attribution_label"].dropna().astype(str).unique().tolist())
                    attr_filter = st.selectbox("Attribution category", attr_options)
                    if attr_filter != "All":
                        view = view[view["opinion_attribution_label"].astype(str) == attr_filter]

                if "contributor_display_name" in view.columns:
                    contributor_options = ["All"] + sorted(view["contributor_display_name"].dropna().astype(str).unique().tolist())
                    contributor_filter = st.selectbox("Contributor", contributor_options)
                    if contributor_filter != "All":
                        view = view[view["contributor_display_name"].astype(str) == contributor_filter]

                display_table(
                    view,
                    [
                        "publication_datetime", "Headline", "Outlet", "Reporter", "contributor_display_name", "contributor_type",
                        "opinion_attribution_label", "opinion_owner", "opinion_owner_type", "stance_owner_side",
                        "attribution_confidence", "attribution_evidence", "dominant_frame", "cluster_display_name", "Link",
                    ],
                    150,
                )
            else:
                st.info("No article-level attribution file found.")
            st.markdown("</div>", unsafe_allow_html=True)

elif page == "Agenda Shapers":
    st.header("Agenda Shapers")
    st.write("Who is shaping the debate, what role are they playing, and where are they positioned?")
    if agenda.empty:
        st.warning("No agenda shaper data found.")
    else:
        group_filter = st.radio("Entity type", ["All", "Actor / organisation", "Media outlet"], horizontal=True)
        view = agenda.copy()
        if group_filter != "All" and "entity_group" in view.columns:
            view = view[view["entity_group"] == group_filter]
        left, right = st.columns([1.45, 1])
        with left:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            plot = view.copy()
            if "avg_stance" not in plot.columns:
                plot["avg_stance"] = 0
            if "impact_score" not in plot.columns:
                plot["impact_score"] = 0
            if "weighted_mentions" not in plot.columns:
                plot["weighted_mentions"] = 1
            fig = px.scatter(
                plot.head(80),
                x="avg_stance",
                y="impact_score",
                size="weighted_mentions",
                color="narrative_role" if "narrative_role" in plot.columns else None,
                hover_name="entity",
                hover_data=[c for c in ["entity_group", "actor_type", "mentions", "unique_stories", "top_cluster", "stance_label"] if c in plot.columns],
                template="plotly_white",
                labels={"avg_stance": "Debate position: preserve ← → move away", "impact_score": "Impact score"},
                size_max=44,
            )
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            fig.update_layout(height=600, margin=dict(l=10, r=10, t=25, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            selected = st.selectbox("Open contributor profile", view["entity"].astype(str).tolist())
            row = view[view["entity"].astype(str) == selected].iloc[0]
            st.subheader(selected)
            html = ""
            for tag in [row.get("entity_group"), row.get("actor_type"), row.get("narrative_role"), row.get("stance_label")]:
                if pd.notna(tag):
                    html += f"<span class='tag'>{tag}</span>"
            st.markdown(html, unsafe_allow_html=True)
            a, b = st.columns(2)
            with a:
                metric_card("Impact score", fmt_num(row.get("impact_score", 0)))
            with b:
                metric_card("Share of voice", f"{float(row.get('share_of_voice', 0)) * 100:.2f}%")
            c, d = st.columns(2)
            with c:
                metric_card("Mentions", fmt_num(row.get("mentions", 0)))
            with d:
                metric_card("Reach", fmt_num(row.get("estimated_reach", 0)))
            st.write(f"**Top narrative:** {row.get('top_cluster', '')}")
            st.write(f"**Top frame:** {row.get('top_frame', '')}")
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Full agenda shaper table")
        display_table(view.sort_values("impact_score", ascending=False) if "impact_score" in view.columns else view, ["entity", "entity_group", "actor_type", "narrative_role", "impact_score", "mentions", "unique_stories", "weighted_mentions", "estimated_reach", "stance_label", "top_cluster", "top_frame"])
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Debate Positioning":
    st.header("Debate Positioning")
    st.write("Maps who is associated with preserving the current position, reforming it, or moving away.")
    if positions.empty:
        st.warning("No debate position table found.")
    else:
        plot = positions.copy()
        if "avg_stance" not in plot.columns:
            plot["avg_stance"] = 0
        if "impact_score" not in plot.columns:
            plot["impact_score"] = 0
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        fig = px.scatter(
            plot,
            x="avg_stance",
            y="impact_score",
            size="weighted_mentions" if "weighted_mentions" in plot.columns else None,
            color="debate_position" if "debate_position" in plot.columns else "stance_label",
            hover_name="actor",
            hover_data=[c for c in ["actor_type", "narrative_role", "top_cluster", "top_frame"] if c in plot.columns],
            template="plotly_white",
            labels={"avg_stance": "Preserve / stay ←     → reform / move away", "impact_score": "Impact score"},
            size_max=44,
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=560, margin=dict(l=10, r=10, t=25, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Position table")
        display_table(plot.sort_values("avg_stance"), ["actor", "actor_type", "debate_position", "avg_stance", "impact_score", "narrative_role", "top_cluster", "top_frame"])
        st.markdown("</div>", unsafe_allow_html=True)
        st.info("Interpretation note: this is the stance of coverage associated with the actor, not a definitive claim about personal belief.")

elif page == "Outlet Influence":
    st.header("Outlet Influence")
    st.write("Which outlets carried the most influence-weighted coverage?")
    if outlets.empty:
        st.warning("No outlet data found.")
    else:
        sort_col = "impact_score" if "impact_score" in outlets.columns else "weighted_mentions"
        view = outlets.sort_values(sort_col, ascending=False)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        fig = px.bar(
            view.head(30),
            x=sort_col,
            y="Outlet",
            color="avg_stance" if "avg_stance" in view.columns else None,
            orientation="h",
            hover_data=[c for c in ["mentions", "unique_stories", "weighted_mentions", "estimated_reach", "top_cluster", "top_frame"] if c in view.columns],
            template="plotly_white",
            labels={sort_col: "Outlet impact", "Outlet": "", "avg_stance": "Avg stance"},
        )
        fig.update_layout(height=720, yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=25, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Outlet table")
        display_table(view, ["Outlet", "impact_score", "mentions", "unique_stories", "weighted_mentions", "estimated_reach", "avg_stance", "avg_sentiment", "top_cluster", "top_frame", "top_tier"])
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Evidence Explorer":
    st.header("Evidence Explorer")
    st.write("Click into the source evidence behind narratives, actors, outlets and scores.")
    if evidence.empty:
        st.warning("No evidence snippets found.")
    else:
        entities = ["All"] + sorted(evidence["entity"].dropna().astype(str).unique().tolist())
        selected = st.selectbox("Entity / outlet", entities)
        ev = evidence.copy()
        if selected != "All":
            ev = ev[ev["entity"].astype(str) == selected]
        if cluster_filter != "All" and "cluster_name" in ev.columns:
            ev = ev[ev["cluster_name"] == cluster_filter]
        st.write(f"Showing **{len(ev)}** evidence snippets.")
        for _, row in ev.head(100).iterrows():
            st.markdown(
                f"""
                <div class="evidence">
                    <b>{row.get("headline", "")}</b><br>
                    <span style="color:#6d827f;font-size:.88rem;">{row.get("outlet", "")} · {row.get("cluster_name", "")} · {row.get("stance_bucket", "")}</span>
                    <p>{row.get("evidence_excerpt", "")}</p>
                    <span class="tag-dark">{row.get("entity", "")}</span>
                    <span class="tag">{row.get("dominant_frame", "")}</span>
                    <br>{source_link(row.get("link", ""))}
                </div>
                """,
                unsafe_allow_html=True,
            )

elif page == "Impact Lab":
    st.header("Impact Lab")
    st.write("Separates current agenda impact from intervention/reframing impact.")
    left, right = st.columns([1, 1])
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("1. Agenda / Media Impact Score")
        st.write("Available now. It answers: **who is most influential in the debate?**")
        st.markdown(
            """
            <div class="formula">Agenda Impact Score =
0.30 × Weighted mentions
+ 0.23 × Estimated reach
+ 0.20 × Unique stories
+ 0.14 × Cluster spread
+ 0.13 × Stance strength</div>
            """,
            unsafe_allow_html=True,
        )
        display_table(agenda.sort_values("impact_score", ascending=False) if "impact_score" in agenda.columns else agenda, ["entity", "entity_group", "narrative_role", "stance_label", "impact_score", "weighted_mentions", "estimated_reach", "unique_stories"], 15)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("2. Intervention Impact Score")
        st.write("Designed for the next data drop. It answers: **did an intervention reframe the conversation afterwards?**")
        st.markdown(
            """
            <div class="formula">Intervention Impact =
0.20 × Reach Shift
+ 0.25 × Narrative Adoption
+ 0.20 × Frame Shift
+ 0.15 × Actor Response
+ 0.10 × Stance Movement
+ 0.10 × Persistence</div>
            """,
            unsafe_allow_html=True,
        )
        if not readiness.empty:
            display_table(readiness, ["matched_articles", "total_articles", "match_rate_pct", "weeks_available", "days_available", "ready_for_trend_analysis", "ready_for_intervention_impact"])
        st.warning("Full intervention scoring needs an intervention/event log with intervention_date, actor, message and type.")
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Methodology":
    st.header("Methodology")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("What the model layer does")
    st.write(
        """
        This demo uses precomputed model outputs. The app itself reads CSVs, which makes it fast and stable.

        1. **Narrative clustering** groups coverage into subtopic clusters.
        2. **Influence weighting** weights coverage by reach, outlet tier, classification and centrality.
        3. **Stance mapping** maps coverage onto a topic-specific debate axis.
        4. **Sentiment/frame separation** distinguishes tone from narrative framing.
        5. **Agenda shaper ranking** identifies actors, organisations and outlets shaping the debate.
        6. **Timeline modelling** tracks narrative volume, stance drift, frame movement and agenda activity through time.
        7. **Evidence drilldown** preserves excerpts and links so each score can be inspected.
        8. **Contributor attribution** separates publication-owned opinion from reported actor opinion, including synthetic demo contributors where byline metadata is missing.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Why this is not just sentiment analysis")
    st.write(
        "Standard sentiment can say coverage is mostly neutral. But neutral tone can still be strategically important. A neutral article can repeatedly connect an issue to security, sovereignty, moral duty, public consent or institutional failure. This app therefore shows stance, frame, role, influence and time movement."
    )
    st.markdown("</div>", unsafe_allow_html=True)
