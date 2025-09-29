import streamlit as st
import os
import re
from decimal import Decimal, InvalidOperation
import math
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
def find_repeating_groups(cols_lower):
    groups = []
    i = 0
    n = len(cols_lower)
    while i < n:
        col = cols_lower[i]
        if "job" in col:
            j = i
            case_idx = None
            comment_idx = None
            for k in range(i+1, min(n, i+6)):
                if "case" in cols_lower[k] or "completion" in cols_lower[k]:
                    case_idx = k
                    for m in range(k+1, min(n, k+6)):
                        if "comment" in cols_lower[m]:
                            comment_idx = m
                            break
                    break
            groups.append((j, case_idx, comment_idx))
            if case_idx:
                i = case_idx + 1
            else:
                i = j + 1
        else:
            i += 1
    return groups

@st.cache_data
def parse_sheet_generic(df_raw, sheet_name):
    rows = []
    cols = list(df_raw.columns)
    cols_lower = [str(c).strip().lower() for c in cols]
    groups = find_repeating_groups(cols_lower)

    # --- handle UFAC wide format ---
    if "ufac" in sheet_name.lower() and len(groups) >= 1:
        for idx, row in df_raw.iterrows():
            name = None
            date = None
            for c, low in zip(cols, cols_lower):
                if "name" in low:
                    name = row.get(c); break
            if name is None and df_raw.shape[1] > 0:
                name = row.iloc[0]
            for c, low in zip(cols, cols_lower):
                if "date" in low:
                    date = row.get(c); break
            if date is None and df_raw.shape[1] > 1:
                date = row.iloc[1]

            for (jidx, cidx, commidx) in groups:
                job_val = row.get(cols[jidx])
                if pd.isna(job_val) or str(job_val).strip() == "":
                    continue
                case_val = row.get(cols[cidx]) if cidx is not None else None
                comment_val = row.get(cols[commidx]) if commidx is not None else None
                rows.append({
                    "Source": sheet_name,
                    "Name": str(name).strip() if pd.notna(name) else None,
                    "Date": date,
                    "JobID": normalize_jobid(job_val),
                    "CaseCompletion": str(case_val).strip().lower() if pd.notna(case_val) else None,
                    "Comments": str(comment_val).strip() if pd.notna(comment_val) else None
                })
        return rows

    # Long format (AA style)
    name_col = None; date_col = None; job_col = None; case_col = None; comment_col = None
    for c, low in zip(cols, cols_lower):
        if name_col is None and "name" in low: name_col = c
        if date_col is None and "date" in low: date_col = c
        if job_col is None and "job" in low and "id" in low: job_col = c
        if case_col is None and ("case" in low or "completion" in low): case_col = c
        if comment_col is None and "comment" in low: comment_col = c

    if job_col is not None:
        for _, row in df_raw.iterrows():
            job_val = row.get(job_col)
            if pd.isna(job_val) or str(job_val).strip() == "":
                continue
            rows.append({
                "Source": sheet_name,
                "Name": str(row.get(name_col)).strip() if name_col and pd.notna(row.get(name_col)) else None,
                "Date": row.get(date_col) if date_col else None,
                "JobID": normalize_jobid(job_val),
                "CaseCompletion": str(row.get(case_col)).strip().lower() if case_col and pd.notna(row.get(case_col)) else None,
                "Comments": str(row.get(comment_col)).strip() if comment_col and pd.notna(row.get(comment_col)) else None
            })
        return rows

    return rows

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
        progress_bar = st.progress(0); status_text = st.empty()

        for i, sheet in enumerate(xls.sheet_names):
            status_text.text(f'Processing sheet: {sheet}')
            try:
                df_raw = pd.read_excel(uploaded_file, sheet_name=sheet, dtype=str, header=0)
            except Exception:
                df_raw = pd.read_excel(uploaded_file, sheet_name=sheet, dtype=str, header=None)
            parsed = parse_sheet_generic(df_raw, sheet)
            all_rows.extend(parsed)
            progress_bar.progress((i + 1) / len(xls.sheet_names))

        progress_bar.empty(); status_text.empty()
        df = pd.DataFrame(all_rows, columns=["Source","Name","Date","JobID","CaseCompletion","Comments"])
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        df["JobID"] = df["JobID"].astype(str).str.strip()
        df = df[~df["JobID"].str.lower().isin({"jobid","casecompletion","completed","comments","none","nan","job id"})]
        df = df[~df["Name"].astype(str).str.lower().isin({"name","nan","none"})]
        df = df.dropna(subset=["JobID"]).reset_index(drop=True)
        df["IsCompleted"] = df.apply(detect_completed_marker, axis=1)
        return df
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

# ----------------- Your chart functions -----------------
# (RE-USE your original chart/metric functions here: create_performance_metrics, 
#  create_agent_performance_chart, create_team_comparison, etc.)
# To keep this message short, I won’t repeat them, but they remain unchanged from your script.

# ---------------- Top/Mid/Low performers ----------------
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
        results.append({"Period": period, "Team": team, "Top": top_df, "Mid": mid_df, "Low": low_df})
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
            for dfb, title, col in zip([s["Top"], s["Mid"], s["Low"]], ["Top performers","Mid performers","Low performers"], [col1,col2,col3]):
                with col:
                    st.markdown(f"**{title}**")
                    if not dfb.empty:
                        st.dataframe(dfb[['Name','Tasks','Completed','CompletionRate']].rename(
                            columns={'Tasks':'Cases','Completed':'Completed','CompletionRate':'% Completed'}))
                    else:
                        st.write("None")

# ---------------- Main App ----------------
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
                    date_range = st.date_input("Date Range", (df["Date"].min().date(), df["Date"].max().date()), min_value=df["Date"].min().date(), max_value=df["Date"].max().date())
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
