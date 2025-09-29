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

    if len(groups) >= 1 and len(groups) > 1 or ("ufac" in sheet_name.lower() and len(groups) >= 1):
        for _, row in df_raw.iterrows():
            name, date = None, None
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
                    "CaseCompletion": str(case_val).strip() if pd.notna(case_val) else None,
                    "Comments": str(comment_val).strip() if pd.notna(comment_val) else None
                })
        return rows

    # Long format
    name_col = date_col = job_col = case_col = comment_col = None
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
                "Name": str(row.get(name_col)).strip() if name_col else None,
                "Date": row.get(date_col),
                "JobID": normalize_jobid(job_val),
                "CaseCompletion": str(row.get(case_col)).strip() if case_col and pd.notna(row.get(case_col)) else None,
                "Comments": str(row.get(comment_col)).strip() if comment_col and pd.notna(row.get(comment_col)) else None
            })
        return rows

    # Fallback
    for _, row in df_raw.iterrows():
        name = row.iloc[0] if df_raw.shape[1] > 0 else None
        date = row.iloc[1] if df_raw.shape[1] > 1 else None
        for val in row.iloc[2:]:
            if pd.isna(val) or str(val).strip() == "": continue
            rows.append({
                "Source": sheet_name,
                "Name": str(name).strip() if pd.notna(name) else None,
                "Date": date,
                "JobID": normalize_jobid(val),
                "CaseCompletion": None,
                "Comments": None
            })
    return rows

@st.cache_data
def detect_completed_marker(row):
    source = row["Source"]
    if "aa" in source.lower():
        s = row["CaseCompletion"]
        if s is None or pd.isna(s): return False
        s_lower = str(s).strip().lower()
        if "completed" in s_lower:
            return "not" not in s_lower and "incomplete" not in s_lower
        return False
    if "ufac" in source.lower():
        jobid = str(row["JobID"]).strip().lower()
        return not jobid.isnumeric()
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

# ---------------- Existing charts ----------------
# (same as your baseline — performance metrics, agent performance, team comparison, etc.)
# ... skipping unchanged chart functions for brevity in this explanation, they stay the same ...

# ---------------- New: Top/Mid/Low performer splits ----------------
def split_agents_by_performance(df, freq="W", rank_by="Tasks"):
    df_with_dates = df.dropna(subset=["Date"])
    if df_with_dates.empty:
        return None
    grouped = df_with_dates.groupby([pd.Grouper(key="Date", freq=freq), "Source", "Name"]).agg(
        Tasks=("JobID", "count"),
        Completed=("IsCompleted", "sum")
    ).reset_index()
    grouped["CompletionRate"] = (grouped["Completed"] / grouped["Tasks"]) * 100

    results = []
    for (period, team), sub in grouped.groupby(["Date", "Source"]):
        sort_col = "Tasks" if rank_by == "Tasks" else "CompletionRate"
        sub = sub.sort_values(sort_col, ascending=False).reset_index(drop=True)
        n = len(sub)
        if n == 0: continue
        results.append({
            "Period": period,
            "Team": team,
            "Top": sub.iloc[:max(1, n//3)],
            "Mid": sub.iloc[n//3:2*n//3],
            "Low": sub.iloc[2*n//3:]
        })
    return results

def display_agent_split_tables(splits, title=""):
    if not splits:
        st.info("No agent data available for Top/Mid/Low analysis.")
        return
    st.subheader(title)
    for s in splits:
        st.markdown(f"**📅 {s['Period'].strftime('%Y-%m-%d')} — Team: {s['Team']}**")
        col1, col2, col3 = st.columns(3)
        with col1: st.markdown("**Top**"); st.dataframe(s["Top"][["Name","Tasks","CompletionRate"]])
        with col2: st.markdown("**Mid**"); st.dataframe(s["Mid"][["Name","Tasks","CompletionRate"]])
        with col3: st.markdown("**Low**"); st.dataframe(s["Low"][["Name","Tasks","CompletionRate"]])

# ---------------- Main Streamlit App ----------------
def main():
    st.markdown('<div class="main-header">📊 GTM Team Performance Dashboard</div>', unsafe_allow_html=True)
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx','xls'])
    if uploaded_file is not None:
        with st.spinner("Processing data..."): df = load_and_process_data(uploaded_file)
        if df is not None:
            with st.sidebar:
                st.header("🔍 Filters")
                teams = ['All'] + sorted(list(df['Source'].unique()))
                selected_team = st.selectbox("Select Team", teams)
                date_range = None
                if not df["Date"].isna().all():
                    date_range = st.date_input("Date Range",
                        value=(df["Date"].min().date(), df["Date"].max().date()),
                        min_value=df["Date"].min().date(), max_value=df["Date"].max().date())
                top_n = st.slider("Top N Agents", 3, 30, 10)
                rank_by = st.radio("Rank Top/Mid/Low By", ["Tasks","CompletionRate"])

            filtered_df = df.copy()
            if selected_team != 'All': filtered_df = filtered_df[filtered_df['Source']==selected_team]
            if date_range and len(date_range)==2:
                filtered_df = filtered_df[(filtered_df['Date'].dt.date>=date_range[0]) &
                                          (filtered_df['Date'].dt.date<=date_range[1])]

            if not filtered_df.empty:
                # Original dashboard content (metrics, charts, etc.)
                st.header("📊 Key Metrics"); create_performance_metrics(filtered_df)
                st.header("👤 Agent Performance")
                st.plotly_chart(create_agent_performance_chart(filtered_df, top_n), use_container_width=True)
                # ... keep rest of your original dashboard logic unchanged ...

                # New section: Top/Mid/Low
                st.header("📊 Top / Mid / Low Performers by Period")
                for freq,label in [("W","Weekly"),("M","Monthly"),("Y","Yearly")]:
                    splits = split_agents_by_performance(filtered_df,freq=freq,rank_by=rank_by)
                    display_agent_split_tables(splits, title=f"🔹 {label} Split")
            else:
                st.warning("No data matches filters.")
    else:
        st.info("👆 Please upload an Excel file to begin analysis")

if __name__=="__main__":
    main()
