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

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="GTM Team Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- STYLING ----------------
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

# ---------------- HELPER FUNCTIONS ----------------
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
    source = row["Source"]
    if "aa" in source.lower():
        s = row["CaseCompletion"]
        if s is None or pd.isna(s):
            return False
        s_lower = str(s).strip().lower()
        if "completed" in s_lower:
            return "not" not in s_lower and "incomplete" not in s_lower
        return False
    if "ufac" in source.lower():
        jobid = str(row["JobID"]).strip().lower()
        if jobid.isnumeric():
            return False
        return True
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

            for _, row in df_raw.iterrows():
                name = row.iloc[0] if df_raw.shape[1] > 0 else None
                date = row.iloc[1] if df_raw.shape[1] > 1 else None
                for val in row.iloc[2:]:
                    if pd.isna(val) or str(val).strip() == "":
                        continue
                    all_rows.append({
                        "Source": sheet,
                        "Name": str(name).strip() if pd.notna(name) else None,
                        "Date": date,
                        "JobID": normalize_jobid(val),
                        "CaseCompletion": None,
                        "Comments": None
                    })

        df = pd.DataFrame(all_rows, columns=["Source","Name","Date","JobID","CaseCompletion","Comments"])
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        df["JobID"] = df["JobID"].astype(str).str.strip()
        df = df.dropna(subset=["JobID"]).reset_index(drop=True)
        df["IsCompleted"] = df.apply(detect_completed_marker, axis=1)
        return df
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

# ---------------- EXISTING CHARTS ----------------
def create_performance_metrics(df):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📋 Total Tasks", f"{len(df):,}", f"{df['IsCompleted'].sum():,} completed")
    with col2:
        unique_jobs = df["JobID"].nunique()
        duplicates = len(df) - unique_jobs
        st.metric("🔢 Unique Jobs", f"{unique_jobs:,}", f"{duplicates:,} duplicates" if duplicates > 0 else "No duplicates")
    with col3:
        st.metric("👥 Active Agents", f"{df['Name'].nunique():,}", f"{len(df['Source'].unique())} teams")
    with col4:
        completion_rate = (df["IsCompleted"].sum() / len(df)) * 100
        st.metric("✅ Completion Rate", f"{completion_rate:.1f}%", f"{df['IsCompleted'].sum():,} of {len(df):,}")

def create_agent_performance_chart(df, top_n=10):
    tasks_per_agent = df.groupby("Name").agg(
        TaskCount=("JobID", "count"),
        Completed=("IsCompleted", "sum")
    ).reset_index().sort_values("TaskCount", ascending=False)
    tasks_per_agent["CompletionRate"] = (tasks_per_agent["Completed"] / tasks_per_agent["TaskCount"]) * 100
    top_agents = tasks_per_agent.head(top_n)

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Task Volume', 'Completion Rate'))
    fig.add_trace(go.Bar(x=top_agents["TaskCount"], y=top_agents["Name"], orientation='h', name="Tasks", marker_color='lightblue'), row=1, col=1)
    fig.add_trace(go.Bar(x=top_agents["CompletionRate"], y=top_agents["Name"], orientation='h', name="Completion %", marker_color='lightgreen'), row=1, col=2)
    fig.update_layout(height=600, showlegend=False, title_text=f"Top {top_n} Agent Performance")
    return fig

def create_team_comparison(df):
    team_stats = df.groupby("Source").agg(
        TotalTasks=("JobID", "count"),
        CompletedTasks=("IsCompleted", "sum"),
        UniqueAgents=("Name", "nunique")
    ).reset_index()
    team_stats["CompletionRate"] = (team_stats["CompletedTasks"] / team_stats["TotalTasks"]) * 100

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{}, {}], [{}, {"type": "domain"}]],
        subplot_titles=('Tasks by Team', 'Completion Rate by Team', 'Agents by Team', 'Task Distribution')
    )
    fig.add_trace(go.Bar(x=team_stats["Source"], y=team_stats["TotalTasks"]), row=1, col=1)
    fig.add_trace(go.Bar(x=team_stats["Source"], y=team_stats["CompletionRate"]), row=1, col=2)
    fig.add_trace(go.Bar(x=team_stats["Source"], y=team_stats["UniqueAgents"]), row=2, col=1)
    fig.add_trace(go.Pie(labels=team_stats["Source"], values=team_stats["TotalTasks"]), row=2, col=2)
    fig.update_layout(height=800, showlegend=False)
    return fig

# ---------------- NEW: TOP/MID/LOW ----------------
def split_agents_by_performance(df, freq="W"):
    df_with_dates = df.dropna(subset=["Date"])
    grouped = df_with_dates.groupby([pd.Grouper(key="Date", freq=freq), "Source", "Name"]).agg(
        TaskCount=("JobID","count"),
        Completed=("IsCompleted","sum")
    ).reset_index()
    grouped["CompletionRate"] = (grouped["Completed"] / grouped["TaskCount"]) * 100

    results = {}
    for (period, team), sub in grouped.groupby(["Date", "Source"]):
        sub_sorted = sub.sort_values("TaskCount", ascending=False).reset_index(drop=True)
        n = len(sub_sorted)
        if n == 0: continue
        top_n = max(1, n//3)
        results[(period, team)] = {
            "Top": sub_sorted.head(top_n),
            "Mid": sub_sorted.iloc[top_n:2*top_n],
            "Low": sub_sorted.iloc[2*top_n:]
        }
    return results

def display_agent_split_tables(splits, label):
    for (period, team), tiers in splits.items():
        st.subheader(f"{label} – {team} – {period.strftime('%Y-%m-%d')}")
        col1, col2, col3 = st.columns(3)
        for col, tier in zip([col1, col2, col3], ["Top","Mid","Low"]):
            with col:
                st.markdown(f"**{tier} Performers**")
                st.dataframe(tiers[tier][["Name","TaskCount","Completed","CompletionRate"]])

# ---------------- MAIN APP ----------------
def main():
    st.markdown('<div class="main-header">📊 GTM Team Performance Dashboard</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx','xls'])
        st.header("🔍 Filters")
        period_option = st.selectbox("Select Period", ["Weekly","Monthly","Yearly"], index=0)
        freq_map = {"Weekly":"W","Monthly":"M","Yearly":"Y"}
        selected_freq = freq_map[period_option]

    if uploaded_file is not None:
        df = load_and_process_data(uploaded_file)
        if df is not None and not df.empty:
            st.header("📊 Key Metrics")
            create_performance_metrics(df)

            st.header("👤 Agent Performance")
            st.plotly_chart(create_agent_performance_chart(df, top_n=10), use_container_width=True)

            st.header("🏆 Team Comparison")
            st.plotly_chart(create_team_comparison(df), use_container_width=True)

            st.header(f"🌟 Top / Mid / Low Performers ({period_option})")
            splits = split_agents_by_performance(df, freq=selected_freq)
            display_agent_split_tables(splits, period_option)

            with st.expander("🔍 View Raw Data"):
                st.dataframe(df.head(200), use_container_width=True)
        else:
            st.warning("No valid records found in uploaded file")
    else:
        st.info("👆 Please upload an Excel file to begin analysis")

if __name__ == "__main__":
    main()
