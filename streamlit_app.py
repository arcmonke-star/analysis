import streamlit as st
import os
import re
from decimal import Decimal, InvalidOperation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# ---------------- Helper functions ----------------
# (all your helper functions stay the same here...)
# normalize_jobid, find_repeating_groups, parse_sheet_generic, detect_completed_marker, load_and_process_data
# create_performance_metrics, create_agent_performance_chart, create_team_comparison, create_time_trends,
# create_duplicate_analysis, create_team_overview, create_team_top_performers, create_team_contribution_pie,
# create_team_trends_small_multiples, create_duplicates_heatmap

# ---------------- Top / Mid / Low functions (new) ----------------
def split_agents_by_performance(df, freq="W", rank_by="Tasks"):
    df_with_dates = df.dropna(subset=["Date"]).copy()
    if df_with_dates.empty:
        return []

    agg = (
        df_with_dates.groupby([pd.Grouper(key="Date", freq=freq), "Source", "Name"])
        .agg(Tasks=("JobID", "count"), Completed=("IsCompleted", "sum"))
        .reset_index()
    )
    agg["CompletionRate"] = (agg["Completed"] / agg["Tasks"]) * 100

    results = []
    for (period, team), group in agg.groupby(["Date", "Source"]):
        g = group.copy().reset_index(drop=True)
        if rank_by == "Tasks":
            g = g.sort_values("Tasks", ascending=False).reset_index(drop=True)
        else:
            g = g.sort_values(["CompletionRate", "Tasks"], ascending=[False, False]).reset_index(drop=True)
        n = len(g)
        if n == 0:
            continue
        top_end = max(1, int(np.ceil(n / 3)))
        mid_end = int(np.ceil(2 * n / 3))
        top_df = g.iloc[:top_end].copy()
        mid_df = g.iloc[top_end:mid_end].copy()
        low_df = g.iloc[mid_end:].copy()
        for dfb in (top_df, mid_df, low_df):
            if not dfb.empty:
                dfb["PeriodStart"] = period
                dfb["Team"] = team
        results.append({
            "Period": period,
            "Team": team,
            "Top": top_df,
            "Mid": mid_df,
            "Low": low_df
        })
    return results

def display_agent_split_tables(splits, label=""):
    if not splits:
        st.info(f"No data for {label} Top/Mid/Low.")
        return
    st.markdown(f"### {label} Top / Mid / Low (by Team & Period)")
    for s in sorted(splits, key=lambda x: (x["Team"], x["Period"])):
        period_str = pd.to_datetime(s["Period"]).strftime("%Y-%m-%d")
        with st.expander(f"{s['Team']} — period starting {period_str}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Top performers**")
                if not s["Top"].empty:
                    st.dataframe(s["Top"][['Name','Tasks','Completed','CompletionRate']].rename(
                        columns={'Tasks':'Cases','Completed':'Completed','CompletionRate':'% Completed'}))
                else:
                    st.write("No Top performers")
            with col2:
                st.markdown("**Mid performers**")
                if not s["Mid"].empty:
                    st.dataframe(s["Mid"][['Name','Tasks','Completed','CompletionRate']].rename(
                        columns={'Tasks':'Cases','Completed':'Completed','CompletionRate':'% Completed'}))
                else:
                    st.write("No Mid performers")
            with col3:
                st.markdown("**Low performers**")
                if not s["Low"].empty:
                    st.dataframe(s["Low"][['Name','Tasks','Completed','CompletionRate']].rename(
                        columns={'Tasks':'Cases','Completed':'Completed','CompletionRate':'% Completed'}))
                else:
                    st.write("No Low performers")

# ---------------- Main Streamlit App ----------------
def main():
    st.markdown('<div class="main-header">📊 GTM Team Performance Dashboard</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx','xls'])

    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        with st.spinner("Processing data..."):
            df = load_and_process_data(uploaded_file)

        if df is not None:
            st.success(f"✅ Processed {len(df):,} records")

            with st.sidebar:
                st.header("🔍 Filters")
                teams = ['All'] + sorted(list(df['Source'].unique()))
                selected_team = st.selectbox("Select Team", teams)
                if not df["Date"].isna().all():
                    date_range = st.date_input(
                        "Date Range",
                        value=(df["Date"].min().date(), df["Date"].max().date()),
                        min_value=df["Date"].min().date(),
                        max_value=df["Date"].max().date()
                    )
                else:
                    date_range = None
                top_n = st.slider("Top N Agents to Show", 3, 30, 10)
                st.markdown("---")
                st.subheader("Top/Mid/Low settings")
                rank_by = st.radio("Rank by:", ["Tasks","CompletionRate"], index=0)
                show_topmidlow = st.checkbox("Show Top / Mid / Low section", value=True)

            filtered_df = df.copy()
            if selected_team != 'All':
                filtered_df = filtered_df[filtered_df['Source']==selected_team]
            if date_range and len(date_range)==2:
                filtered_df = filtered_df[(filtered_df['Date'].dt.date>=date_range[0]) & (filtered_df['Date'].dt.date<=date_range[1])]

            if not filtered_df.empty:
                st.header("📊 Key Metrics"); create_performance_metrics(filtered_df)
                st.header("👤 Agent Performance"); st.plotly_chart(create_agent_performance_chart(filtered_df, top_n), use_container_width=True)

                if selected_team=='All':
                    st.header("🏆 Team Comparison"); st.plotly_chart(create_team_comparison(filtered_df), use_container_width=True)
                    st.subheader("Team Completion Rates"); overview_fig, team_stats=create_team_overview(filtered_df); st.plotly_chart(overview_fig,use_container_width=True); st.dataframe(team_stats)
                    st.subheader("🌟 Top Performers per Team"); fig=create_team_top_performers(filtered_df,top_n=min(top_n,20));
                    if fig: st.plotly_chart(fig,use_container_width=True)
                    st.subheader("📌 Team Contribution"); st.plotly_chart(create_team_contribution_pie(filtered_df),use_container_width=True)
                    st.subheader("📈 Team Trends (small multiples)"); small=create_team_trends_small_multiples(filtered_df)
                    if small: st.plotly_chart(small,use_container_width=True)
                    st.subheader("🔍 Duplicate JobID Heatmap"); dup_heat=create_duplicates_heatmap(filtered_df,20)
                    if dup_heat: st.plotly_chart(dup_heat,use_container_width=True)
                else:
                    st.subheader(f"Team: {selected_team} — Breakdown"); single_team_top=create_team_top_performers(filtered_df,top_n)
                    if single_team_top: st.plotly_chart(single_team_top,use_container_width=True)
                    st.subheader("Team Trends Over Time"); team_trend=create_time_trends(filtered_df)
                    if team_trend: st.plotly_chart(team_trend,use_container_width=True)
                    st.subheader("Duplicate Analysis (Team)"); dup=create_duplicate_analysis(filtered_df)
                    if dup:
                        dup_chart,dup_data=dup; st.plotly_chart(dup_chart,use_container_width=True); st.dataframe(dup_data)

                if show_topmidlow:
                    st.markdown("---"); st.header("📊 Top / Mid / Low Performers by Period (per Team)")
                    for freq,label in [("W","Weekly"),("M","Monthly"),("Y","Yearly")]:
                        splits=split_agents_by_performance(filtered_df,freq=freq,rank_by=rank_by)
                        display_agent_split_tables(splits,label=label)

                with st.expander("🔍 View Raw Data"):
                    st.dataframe(filtered_df.head(200),use_container_width=True)
                    if len(filtered_df)>200: st.info(f"Showing first 200 rows of {len(filtered_df):,} total records")
            else:
                st.warning("No data matches the current filters.")
    else:
        st.info("👆 Please upload an Excel file to begin analysis")
        with st.expander("📋 Expected Data Format"):
            st.markdown("Expected sheet formats for UFAC and AA data with Name, Date, JobID, CaseCompletion etc.")

if __name__=="__main__":
    main()
