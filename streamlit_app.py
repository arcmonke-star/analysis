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

    if len(groups) >= 1 and len(groups) > 1 or ("ufac" in sheet_name.lower() and len(groups) >= 1):
        for idx, row in df_raw.iterrows():
            name = None
            date = None
            for c, low in zip(cols, cols_lower):
                if "name" in low:
                    name = row.get(c)
                    break
            if name is None:
                name = row.iloc[0] if df_raw.shape[1] > 0 else None
            for c, low in zip(cols, cols_lower):
                if "date" in low:
                    date = row.get(c)
                    break
            if date is None and df_raw.shape[1] > 1:
                date = row.iloc[1]

            for (jidx, cidx, commidx) in groups:
                try:
                    job_col = cols[jidx]
                    job_val = row.get(job_col)
                except Exception:
                    job_val = None
                if pd.isna(job_val) or str(job_val).strip() == "":
                    continue
                case_val = None
                if cidx is not None:
                    case_col = cols[cidx]
                    case_val = row.get(case_col)
                comment_val = None
                if commidx is not None:
                    comment_col = cols[commidx]
                    comment_val = row.get(comment_col)
                rows.append({
                    "Source": sheet_name,
                    "Name": str(name).strip() if pd.notna(name) else None,
                    "Date": date,
                    "JobID": normalize_jobid(job_val),
                    "CaseCompletion": str(case_val).strip() if pd.notna(case_val) else None,
                    "Comments": str(comment_val).strip() if pd.notna(comment_val) else None
                })
        return rows

    # Long format detection
    name_col = None; date_col = None; job_col = None; case_col = None; comment_col = None
    for c, low in zip(cols, cols_lower):
        if name_col is None and "name" in low:
            name_col = c
        if date_col is None and "date" in low:
            date_col = c
        if job_col is None and "job" in low and "id" in low:
            job_col = c
        if case_col is None and ("case" in low or "completion" in low):
            case_col = c
        if comment_col is None and "comment" in low:
            comment_col = c

    if job_col is not None:
        for _, row in df_raw.iterrows():
            job_val = row.get(job_col)
            if pd.isna(job_val) or str(job_val).strip() == "":
                continue
            name = row.get(name_col) if name_col is not None else None
            date = row.get(date_col) if date_col is not None else None
            case_val = row.get(case_col) if case_col is not None else None
            comment_val = row.get(comment_col) if comment_col is not None else None
            rows.append({
                "Source": sheet_name,
                "Name": str(name).strip() if pd.notna(name) else None,
                "Date": date,
                "JobID": normalize_jobid(job_val),
                "CaseCompletion": str(case_val).strip() if pd.notna(case_val) else None,
                "Comments": str(comment_val).strip() if pd.notna(comment_val) else None
            })
        return rows

    # Fallback
    for _, row in df_raw.iterrows():
        name = row.iloc[0] if df_raw.shape[1] > 0 else None
        date = row.iloc[1] if df_raw.shape[1] > 1 else None
        for val in row.iloc[2:]:
            if pd.isna(val) or str(val).strip() == "":
                continue
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
    """Load and process the uploaded Excel file"""
    try:
        xls = pd.ExcelFile(uploaded_file)
        all_rows = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i, sheet in enumerate(xls.sheet_names):
            status_text.text(f'Processing sheet: {sheet}')
            try:
                df_raw = pd.read_excel(uploaded_file, sheet_name=sheet, dtype=str, header=0)
            except Exception:
                df_raw = pd.read_excel(uploaded_file, sheet_name=sheet, dtype=str, header=None)
            parsed = parse_sheet_generic(df_raw, sheet)
            all_rows.extend(parsed)
            progress_bar.progress((i + 1) / len(xls.sheet_names))
        progress_bar.empty()
        status_text.empty()
        # Build DataFrame
        df = pd.DataFrame(all_rows, columns=["Source","Name","Date","JobID","CaseCompletion","Comments"])
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        df["JobID"] = df["JobID"].astype(str).str.strip()
        # Clean data
        df = df[~df["JobID"].str.lower().isin({"jobid","casecompletion","completed","comments","none","nan","job id"})]
        df = df[~df["Name"].astype(str).str.lower().isin({"name","nan","none"})]
        df = df.dropna(subset=["JobID"]).reset_index(drop=True)
        # Add completion marker
        df["IsCompleted"] = df.apply(detect_completed_marker, axis=1)

        # Apply GTM AA cap: max 12 cases per agent per day
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

# ---------------- Existing charts (updated to use CappedCount) ----------------
def create_performance_metrics(df):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="📋 Total Tasks",
            value=f"{df['CappedCount'].sum():,}",
            delta=f"{df['IsCompleted'].sum():,} completed"
        )
    with col2:
        unique_jobs = df["JobID"].nunique()
        duplicates = len(df) - unique_jobs
        st.metric(
            label="🔢 Unique Jobs",
            value=f"{unique_jobs:,}",
            delta=f"{duplicates:,} duplicates" if duplicates > 0 else "No duplicates"
        )
    with col3:
        st.metric(
            label="👥 Active Agents",
            value=f"{df['Name'].nunique():,}",
            delta=f"{len(df['Source'].unique())} teams"
        )
    with col4:
        completion_rate = (df["IsCompleted"].sum() / df["CappedCount"].sum()) * 100
        st.metric(
            label="✅ Completion Rate",
            value=f"{completion_rate:.1f}%",
            delta=f"{df['IsCompleted'].sum():,} of {df['CappedCount'].sum():,}"
        )

# ---------------
# The rest of your charts (agent, team, trends, duplicates, top/mid/low) should all replace
# ("JobID", "count") with ("CappedCount", "sum") in groupby aggregations.
# ---------------

# ---------------- Main Streamlit App ----------------
def main():
    st.markdown('<div class="main-header">📊 GTM Team Performance Dashboard</div>', unsafe_allow_html=True)
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx', 'xls'])
    if uploaded_file is not None:
        with st.spinner("Processing data..."):
            df = load_and_process_data(uploaded_file)
        if df is not None:
            st.success(f"✅ Processed {len(df):,} records")
            st.header("📊 Key Metrics")
            create_performance_metrics(df)
            # ... keep the rest of your dashboard unchanged, using updated groupby logic ...
    else:
        st.info("👆 Please upload an Excel file to begin analysis")

if __name__ == "__main__":
    main()
