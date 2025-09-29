import streamlit as st
import os
import re
from decimal import Decimal, InvalidOperation
import math
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

# Custom CSS
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


# ---------------- Helper functions ----------------
@st.cache_data
def normalize_jobid(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    s = s.replace(",", "").replace(" ", "")
    try:
        if re.search(r'[eE]', s) or re.match(r'^\d+\.\d+$', s):
            d = Decimal(s)
            s_full = format(d.normalize(), 'f')
            if '.' in s_full:
                s_full = s_full.split('.')[0]
            return s_full
    except (InvalidOperation, ValueError):
        pass
    s = re.sub(r'\.0+$', '', s)
    return s


@st.cache_data
def detect_completed_marker(row):
    source = row["Source"].lower()
    s = str(row.get("CaseCompletion", "")).strip().lower()

    if "aa" in source:
        return s.startswith("completed") and "not" not in s and "incomplete" not in s
    if "ufac" in source:
        return s == "completed"
    return False


@st.cache_data
def load_and_process_data(uploaded_file):
    try:
        xls = pd.ExcelFile(uploaded_file)
        all_rows = []
        for sheet in xls.sheet_names:
            try:
                df_raw = pd.read_excel(uploaded_file, sheet_name=sheet, dtype=str, header=0)
            except Exception:
                df_raw = pd.read_excel(uploaded_file, sheet_name=sheet, dtype=str, header=None)
            # very simplified parsing for brevity
            for _, row in df_raw.iterrows():
                if row.isna().all():
                    continue
                all_rows.append({
                    "Source": sheet,
                    "Name": row.iloc[0],
                    "Date": row.iloc[1] if len(row) > 1 else None,
                    "JobID": normalize_jobid(row.iloc[2]) if len(row) > 2 else None,
                    "CaseCompletion": row.iloc[3] if len(row) > 3 else None,
                    "Comments": row.iloc[4] if len(row) > 4 else None
                })

        df = pd.DataFrame(all_rows, columns=["Source", "Name", "Date", "JobID", "CaseCompletion", "Comments"])
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        df = df.dropna(subset=["JobID"]).reset_index(drop=True)

        # Completion flag
        df["IsCompleted"] = df.apply(detect_completed_marker, axis=1)

        # --- Apply GTM AA daily cap: max 12 cases per agent per day
        df["CappedCount"] = 1
        aa_mask = df["Source"].str.contains("aa", case=False, na=False)
        df.loc[aa_mask, "CapRank"] = (
            df[aa_mask]
            .sort_values(["Date", "Name"])
            .groupby([df["Date"], df["Name"]])
            .cumcount() + 1
        )
        df.loc[aa_mask & (df["CapRank"] > 12), "CappedCount"] = 0
        df.drop(columns=["CapRank"], inplace=True)

        return df
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None


# ---------------- Chart Functions ----------------
def create_performance_metrics(df):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📋 Total Tasks", f"{df['CappedCount'].sum():,}", delta=f"{df['IsCompleted'].sum():,} completed")
    with col2:
        unique_jobs = df["JobID"].nunique()
        duplicates = len(df) - unique_jobs
        st.metric("🔢 Unique Jobs", f"{unique_jobs:,}", delta=f"{duplicates:,} duplicates" if duplicates > 0 else "No duplicates")
    with col3:
        st.metric("👥 Active Agents", f"{df['Name'].nunique():,}", delta=f"{len(df['Source'].unique())} teams")
    with col4:
        completion_rate = (df["IsCompleted"].sum() / df["CappedCount"].sum()) * 100 if df["CappedCount"].sum() > 0 else 0
        st.metric("✅ Completion Rate", f"{completion_rate:.1f}%", delta=f"{df['IsCompleted'].sum():,} of {df['CappedCount'].sum():,}")


def create_agent_performance_chart(df, top_n=10):
    tasks_per_agent = df.groupby("Name").agg(
        TaskCount=("CappedCount", "sum"),
        Completed=("IsCompleted", "sum")
    ).reset_index().sort_values("TaskCount", ascending=False)
    tasks_per_agent["CompletionRate"] = (tasks_per_agent["Completed"] / tasks_per_agent["TaskCount"]) * 100
    top_agents = tasks_per_agent.head(top_n)

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Task Volume', 'Completion Rate'))
    fig.add_trace(go.Bar(x=top_agents["TaskCount"], y=top_agents["Name"], orientation='h', name="Tasks"), row=1, col=1)
    fig.add_trace(go.Bar(x=top_agents["CompletionRate"], y=top_agents["Name"], orientation='h', name="Completion %"), row=1, col=2)
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def create_team_comparison(df):
    team_stats = df.groupby("Source").agg(
        TotalTasks=("CappedCount", "sum"),
        CompletedTasks=("IsCompleted", "sum"),
        UniqueAgents=("Name", "nunique")
    ).reset_index()
    team_stats["CompletionRate"] = (team_stats["CompletedTasks"] / team_stats["TotalTasks"]) * 100

    fig = make_subplots(rows=2, cols=2, subplot_titles=('Tasks by Team', 'Completion Rate by Team', 'Agents by Team', 'Task Distribution'))
    fig.add_trace(go.Bar(x=team_stats["Source"], y=team_stats["TotalTasks"]), row=1, col=1)
    fig.add_trace(go.Bar(x=team_stats["Source"], y=team_stats["CompletionRate"]), row=1, col=2)
    fig.add_trace(go.Bar(x=team_stats["Source"], y=team_stats["UniqueAgents"]), row=2, col=1)
    fig.add_trace(go.Pie(labels=team_stats["Source"], values=team_stats["TotalTasks"]), row=2, col=2)
    fig.update_layout(height=800, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def create_time_trends(df):
    if df["Date"].isna().all():
        st.warning("No valid dates for trends")
        return
    daily_stats = df.groupby([df["Date"].dt.date, "Source"]).agg(
        DailyTasks=("CappedCount", "sum"),
        DailyCompleted=("IsCompleted", "sum")
    ).reset_index()
    daily_stats["CompletionRate"] = (daily_stats["DailyCompleted"] / daily_stats["DailyTasks"]) * 100

    fig = make_subplots(rows=2, cols=1, subplot_titles=('Daily Task Volume', 'Completion Rate'), shared_xaxes=True)
    for source in daily_stats["Source"].unique():
        sd = daily_stats[daily_stats["Source"] == source]
        fig.add_trace(go.Scatter(x=sd["Date"], y=sd["DailyTasks"], mode='lines+markers', name=f"{source} Tasks"), row=1, col=1)
        fig.add_trace(go.Scatter(x=sd["Date"], y=sd["CompletionRate"], mode='lines+markers', name=f"{source} %"), row=2, col=1)
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)


def create_duplicate_analysis(df):
    dup_counts = df.groupby("JobID").size().reset_index(name="Count")
    duplicates = dup_counts[dup_counts["Count"] > 1]
    if duplicates.empty:
        st.info("No duplicate JobIDs found")
        return
    top_dups = duplicates.sort_values("Count", ascending=False).head(10)
    fig = px.bar(top_dups, x="Count", y="JobID", orientation='h', title="Top 10 Duplicates")
    st.plotly_chart(fig, use_container_width=True)


# ----------- Top / Mid / Low performers ------------
def split_agents_by_performance(df, freq="W"):
    df_with_dates = df.dropna(subset=["Date"]).copy()
    agg = df_with_dates.groupby([pd.Grouper(key="Date", freq=freq), "Source", "Name"]).agg(
        Tasks=("CappedCount", "sum"),
        Completed=("IsCompleted", "sum")
    ).reset_index()
    agg["CompletionRate"] = (agg["Completed"] / agg["Tasks"]) * 100
    results = []
    for (period, team), group in agg.groupby(["Date", "Source"]):
        g = group.sort_values("Tasks", ascending=False).reset_index(drop=True)
        n = len(g)
        if n == 0: continue
        top = g.iloc[:math.ceil(n/3)]
        mid = g.iloc[math.ceil(n/3):math.ceil(2*n/3)]
        low = g.iloc[math.ceil(2*n/3):]
        results.append({"Period": period, "Team": team, "Top": top, "Mid": mid, "Low": low})
    return results


def display_agent_split_tables(splits, label=""):
    for s in splits:
        with st.expander(f"{s['Team']} — period {s['Period']}"):
            for tier, df_tier in [("Top", s["Top"]), ("Mid", s["Mid"]), ("Low", s["Low"])]:
                st.write(f"{tier} performers")
                if not df_tier.empty:
                    show = df_tier[["Name", "Tasks", "Completed", "CompletionRate"]].copy()
                    show["CompletionRate"] = show["CompletionRate"].round(1)
                    st.dataframe(show)
                else:
                    st.write("No data")


# ---------------- Main App ----------------
def main():
    st.markdown('<div class="main-header">📊 GTM Team Performance Dashboard</div>', unsafe_allow_html=True)
    with st.sidebar:
        st.header("📁 Upload File")
        uploaded_file = st.file_uploader("Upload Excel", type=['xlsx', 'xls'])

    if uploaded_file is not None:
        df = load_and_process_data(uploaded_file)
        if df is not None and not df.empty:
            st.header("📊 Key Metrics")
            create_performance_metrics(df)

            st.header("👤 Agent Performance")
            create_agent_performance_chart(df)

            st.header("🏆 Team Comparison")
            create_team_comparison(df)

            st.header("📈 Trends")
            create_time_trends(df)

            st.header("🔍 Duplicate Analysis")
            create_duplicate_analysis(df)

            st.header("⭐ Top/Mid/Low Performers")
            splits = split_agents_by_performance(df, freq="W")
            display_agent_split_tables(splits, "Weekly")
        else:
            st.warning("No valid records after processing.")
    else:
        st.info("Please upload a file to start.")


if __name__ == "__main__":
    main()
