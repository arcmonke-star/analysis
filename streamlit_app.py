import streamlit as st
import os
import re
from decimal import Decimal, InvalidOperation
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="GTM Team Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Helper functions (same as your code) ----------------
# normalize_jobid, find_repeating_groups, parse_sheet_generic,
# detect_completed_marker, load_and_process_data, etc...
# ---------------- Existing charts (same as your code) ----------------
# create_performance_metrics, create_agent_performance_chart, create_team_comparison,
# create_time_trends, create_duplicate_analysis, create_team_overview, etc.

# ---------------- Period-based insights ----------------
def aggregate_by_period(df, period="W"):
    df_with_dates = df.dropna(subset=["Date"])
    if df_with_dates.empty:
        return None
    grouped = df_with_dates.groupby([pd.Grouper(key="Date", freq=period), "Source"]).agg(
        Tasks=("JobID", "count"),
        Completed=("IsCompleted", "sum")
    ).reset_index()
    grouped["CompletionRate"] = (grouped["Completed"] / grouped["Tasks"]) * 100
    return grouped

def create_period_trends(df, granularity="W"):
    period_map = {"W": "Weekly", "M": "Monthly", "Y": "Yearly"}
    grouped = aggregate_by_period(df, granularity)
    if grouped is None:
        st.info("No valid dated records available for period trends.")
        return None

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=(f"{period_map[granularity]} Task Volume", f"{period_map[granularity]} Completion Rate")
    )
    for source in grouped["Source"].unique():
        source_data = grouped[grouped["Source"] == source]
        fig.add_trace(go.Bar(x=source_data["Date"], y=source_data["Tasks"], name=f"{source} Tasks"), row=1, col=1)
        fig.add_trace(go.Scatter(x=source_data["Date"], y=source_data["CompletionRate"],
                                 mode="lines+markers", name=f"{source} %"), row=2, col=1)
    fig.update_layout(height=600, title_text=f"{period_map[granularity]} Trends")
    return fig

def split_agents_by_performance(df, granularity="W"):
    df_with_dates = df.dropna(subset=["Date"])
    if df_with_dates.empty:
        return None
    period_groups = df_with_dates.groupby([pd.Grouper(key="Date", freq=granularity), "Source", "Name"]).agg(
        Tasks=("JobID", "count"),
        Completed=("IsCompleted", "sum")
    ).reset_index()
    period_groups["CompletionRate"] = (period_groups["Completed"] / period_groups["Tasks"]) * 100
    results = []
    for (period, source), sub in period_groups.groupby(["Date", "Source"]):
        sub = sub.sort_values("Tasks", ascending=False).reset_index(drop=True)
        n = len(sub)
        if n == 0: continue
        top = sub.iloc[:max(1, n // 3)]
        mid = sub.iloc[n // 3:2 * n // 3]
        low = sub.iloc[2 * n // 3:]
        results.append({"Period": period, "Team": source, "Top": top, "Mid": mid, "Low": low})
    return results

def display_agent_split_tables(splits):
    if not splits:
        st.info("No agent data available for splits.")
        return
    for s in splits:
        st.subheader(f"📅 {s['Period'].strftime('%Y-%m-%d')} — Team: {s['Team']}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Top performers**")
            st.dataframe(s["Top"][["Name", "Tasks", "CompletionRate"]])
        with col2:
            st.markdown("**Mid performers**")
            st.dataframe(s["Mid"][["Name", "Tasks", "CompletionRate"]])
        with col3:
            st.markdown("**Low performers**")
            st.dataframe(s["Low"][["Name", "Tasks", "CompletionRate"]])

# ---------------- Main Streamlit App ----------------
def main():
    st.markdown('<div class="main-header">📊 GTM Team Performance Dashboard</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx', 'xls'])

    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        with st.spinner("Processing data..."):
            df = load_and_process_data(uploaded_file)

        if df is not None:
            tabs = st.tabs(["📊 Dashboard", "📅 Period Analysis"])

            # -------- TAB 1: Dashboard (your original content) --------
            with tabs[0]:
                st.header("📊 Key Metrics")
                create_performance_metrics(df)

                st.header("👤 Agent Performance")
                agent_chart = create_agent_performance_chart(df, top_n=10)
                st.plotly_chart(agent_chart, use_container_width=True)

                st.header("🏆 Team Comparison")
                team_chart = create_team_comparison(df)
                st.plotly_chart(team_chart, use_container_width=True)

                st.header("📈 Trends Over Time")
                time_trend_fig = create_time_trends(df)
                if time_trend_fig:
                    st.plotly_chart(time_trend_fig, use_container_width=True)

                st.header("🔍 Duplicate Analysis")
                dup_result = create_duplicate_analysis(df)
                if dup_result:
                    dup_chart, dup_data = dup_result
                    st.plotly_chart(dup_chart, use_container_width=True)
                    with st.expander("View Duplicate Details"):
                        st.dataframe(dup_data)

                with st.expander("🔍 View Raw Data"):
                    st.dataframe(df.head(200), use_container_width=True)

            # -------- TAB 2: Period Analysis --------
            with tabs[1]:
                st.header("📅 Period-Based Analysis")
                granularity = st.selectbox("Select Granularity", {"W": "Weekly", "M": "Monthly", "Y": "Yearly"})
                fig = create_period_trends(df, granularity=granularity)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                splits = split_agents_by_performance(df, granularity=granularity)
                display_agent_split_tables(splits)

    else:
        st.info("👆 Please upload an Excel file to begin analysis")

if __name__ == "__main__":
    main()
