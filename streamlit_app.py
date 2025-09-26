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
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
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

# Helper functions
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
        
        return df
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def create_performance_metrics(df):
    """Create performance metrics cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📋 Total Tasks",
            value=f"{len(df):,}",
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
        completion_rate = (df["IsCompleted"].sum() / len(df)) * 100
        st.metric(
            label="✅ Completion Rate",
            value=f"{completion_rate:.1f}%",
            delta=f"{df['IsCompleted'].sum():,} of {len(df):,}"
        )

def create_agent_performance_chart(df, top_n=10):
    """Create interactive agent performance chart"""
    tasks_per_agent = df.groupby("Name").agg(
        TaskCount=("JobID", "count"),
        Completed=("IsCompleted", "sum")
    ).reset_index().sort_values("TaskCount", ascending=False)
    
    tasks_per_agent["CompletionRate"] = (tasks_per_agent["Completed"] / tasks_per_agent["TaskCount"]) * 100
    
    top_agents = tasks_per_agent.head(top_n)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Task Volume', 'Completion Rate'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Task volume chart
    fig.add_trace(
        go.Bar(
            x=top_agents["TaskCount"],
            y=top_agents["Name"],
            orientation='h',
            name="Tasks",
            marker_color='lightblue',
            text=top_agents["TaskCount"],
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # Completion rate chart
    fig.add_trace(
        go.Bar(
            x=top_agents["CompletionRate"],
            y=top_agents["Name"],
            orientation='h',
            name="Completion %",
            marker_color='lightgreen',
            text=[f"{rate:.1f}%" for rate in top_agents["CompletionRate"]],
            textposition='outside'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text=f"Top {top_n} Agent Performance"
    )
    
    return fig

def create_team_comparison(df):
    """Create team comparison charts"""
    team_stats = df.groupby("Source").agg(
        TotalTasks=("JobID", "count"),
        CompletedTasks=("IsCompleted", "sum"),
        UniqueAgents=("Name", "nunique")
    ).reset_index()
    
    team_stats["CompletionRate"] = (team_stats["CompletedTasks"] / team_stats["TotalTasks"]) * 100
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Tasks by Team', 'Completion Rate by Team', 'Agents by Team', 'Task Distribution'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "pie"}]]
    )
    
    # Tasks by team
    fig.add_trace(
        go.Bar(x=team_stats["Source"], y=team_stats["TotalTasks"], 
               name="Total Tasks", marker_color='skyblue'),
        row=1, col=1
    )
    
    # Completion rate by team
    fig.add_trace(
        go.Bar(x=team_stats["Source"], y=team_stats["CompletionRate"],
               name="Completion Rate", marker_color='lightcoral'),
        row=1, col=2
    )
    
    # Agents by team
    fig.add_trace(
        go.Bar(x=team_stats["Source"], y=team_stats["UniqueAgents"],
               name="Unique Agents", marker_color='lightgreen'),
        row=2, col=1
    )
    
    # Pie chart for task distribution
    fig.add_trace(
        go.Pie(labels=team_stats["Source"], values=team_stats["TotalTasks"],
               name="Task Distribution"),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False)
    return fig

def create_time_trends(df):
    """Create time-based trend analysis"""
    if df["Date"].isna().all():
        st.warning("No valid dates found for trend analysis")
        return None
    
    # Filter out NaT dates and create daily aggregation
    df_with_dates = df.dropna(subset=["Date"])
    daily_stats = df_with_dates.groupby([df_with_dates["Date"].dt.date, "Source"]).agg(
        DailyTasks=("JobID", "count"),
        DailyCompleted=("IsCompleted", "sum")
    ).reset_index()
    
    daily_stats["CompletionRate"] = (daily_stats["DailyCompleted"] / daily_stats["DailyTasks"]) * 100
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Daily Task Volume by Team', 'Daily Completion Rate by Team'),
        shared_xaxes=True
    )
    
    for source in daily_stats["Source"].unique():
        source_data = daily_stats[daily_stats["Source"] == source]
        
        fig.add_trace(
            go.Scatter(x=source_data["Date"], y=source_data["DailyTasks"],
                      mode='lines+markers', name=f'{source} - Tasks'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=source_data["Date"], y=source_data["CompletionRate"],
                      mode='lines+markers', name=f'{source} - Completion %'),
            row=2, col=1
        )
    
    fig.update_layout(height=600, title_text="Trends Over Time")
    return fig

def create_duplicate_analysis(df):
    """Analyze duplicate JobIDs"""
    dup_counts = df.groupby("JobID").size().reset_index(name="Count")
    duplicates = dup_counts[dup_counts["Count"] > 1].sort_values("Count", ascending=False)
    
    if duplicates.empty:
        st.info("No duplicate JobIDs found!")
        return None
    
    # Top duplicates chart
    top_dups = duplicates.head(10)
    fig = px.bar(
        top_dups, 
        x="Count", 
        y="JobID", 
        orientation='h',
        title="Top 10 Most Duplicated JobIDs",
        color="Count",
        color_continuous_scale="Reds"
    )
    fig.update_layout(height=400)
    
    return fig, duplicates

# Main Streamlit App
def main():
    st.markdown('<div class="main-header">📊 GTM Team Performance Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar for file upload and controls
    # Main Streamlit App
def main():
    st.markdown('<div class="main-header">📊 GTM Team Performance Dashboard</div>', unsafe_allow_html=True)

    # Sidebar: file upload
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload Excel File",
            type=['xlsx', 'xls'],
            help="Upload your GTM team performance Excel file"
        )

    # If a file is uploaded
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")

        # Load and process
        with st.spinner("Processing data..."):
            df = load_and_process_data(uploaded_file)

        if df is not None:
            st.success(f"✅ Processed {len(df):,} records")

            # Sidebar filters
            with st.sidebar:
                st.header("🔍 Filters")

                # Team filter
                teams = ['All'] + list(df['Source'].unique())
                selected_team = st.selectbox("Select Team", teams)

                # Date filter
                if not df["Date"].isna().all():
                    date_range = st.date_input(
                        "Date Range",
                        value=(df["Date"].min().date(), df["Date"].max().date()),
                        min_value=df["Date"].min().date(),
                        max_value=df["Date"].max().date()
                    )
                else:
                    date_range = None

                # Top N agents
                top_n = st.slider("Top N Agents to Show", 5, 30, 10)

            # Apply filters
            filtered_df = df.copy()
            if selected_team != 'All':
                filtered_df = filtered_df[filtered_df['Source'] == selected_team]
            if date_range and len(date_range) == 2:
                filtered_df = filtered_df[
                    (filtered_df['Date'].dt.date >= date_range[0]) &
                    (filtered_df['Date'].dt.date <= date_range[1])
                ]

            # ----------------- MAIN DASHBOARD -----------------
            if not filtered_df.empty:
                # Metrics
                st.header("📊 Key Metrics")
                create_performance_metrics(filtered_df)

                # Agent performance
                st.header("👤 Agent Performance")
                agent_chart = create_agent_performance_chart(filtered_df, top_n)
                st.plotly_chart(agent_chart, use_container_width=True)

                # Team comparison (only if all teams selected)
                if selected_team == 'All':
                    st.header("🏆 Team Comparison")
                    team_chart = create_team_comparison(filtered_df)
                    st.plotly_chart(team_chart, use_container_width=True)

                # Trends
                st.header("📈 Time Trends")
                trend_chart = create_time_trends(filtered_df)
                if trend_chart:
                    st.plotly_chart(trend_chart, use_container_width=True)

                # Duplicates
                st.header("🔍 Duplicate Analysis")
                dup_result = create_duplicate_analysis(filtered_df)
                if dup_result:
                    dup_chart, dup_data = dup_result
                    st.plotly_chart(dup_chart, use_container_width=True)
                    with st.expander("View Duplicate Details"):
                        st.dataframe(dup_data)

                # Raw data view
                with st.expander("🔍 View Raw Data"):
                    st.dataframe(filtered_df.head(100), use_container_width=True)
                    if len(filtered_df) > 100:
                        st.info(f"Showing first 100 rows of {len(filtered_df):,} total records")

            else:
                st.warning("No data matches the current filters.")

    # If no file is uploaded
    else:
        st.info("👆 Please upload an Excel file to begin analysis")

        # Show expected data format
        with st.expander("📋 Expected Data Format"):
            st.markdown("""
            **Your Excel file should contain sheets with:**

            **For UFAC format (wide):**
            - Name, Date columns
            - Repeating job/case/comment columns

            **For AA format (long):**
            - Name, Date, JobID, CaseCompletion columns

            **Supported sheet naming:**
            - GTM AA, GTM UFAC, etc.
            """)

if __name__ == "__main__":
    main()
