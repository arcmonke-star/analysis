import streamlit as st
import re
from decimal import Decimal, InvalidOperation
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

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
    .main-header { font-size: 2.4rem; font-weight: bold; text-align: center; color: #1f77b4; margin-bottom: 1.5rem; }
    .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4; }
    .stMetric { background-color: white; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# ---------------- HELPERS / PARSING ----------------
@st.cache_data
def normalize_jobid(val):
    """Normalize job id values (handle scientific notation, trailing .0, commas etc.)"""
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    s = s.replace(",", "").replace(" ", "")
    try:
        # If in scientific or decimal form, convert to plain integer-like string
        if re.search(r'[eE]', s) or re.match(r'^\d+\.\d+$', s):
            d = Decimal(s)
            s_full = format(d.normalize(), 'f')
            if '.' in s_full:
                s_full = s_full.split('.')[0]
            return s_full
    except (InvalidOperation, ValueError):
        pass
    # remove trailing .0
    s = re.sub(r'\.0+$', '', s)
    return s

@st.cache_data
def find_repeating_groups(cols_lower):
    """Detect repeating job / case / comment groups (for UFAC/wide)."""
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
    """
    Parse both wide (UFAC) and long (AA) style sheets into rows:
    returns list of dict rows with columns: Source, Name, Date, JobID, CaseCompletion, Comments
    """
    rows = []
    cols = list(df_raw.columns)
    cols_lower = [str(c).strip().lower() for c in cols]
    groups = find_repeating_groups(cols_lower)

    # If a wide format with repeating groups (UFAC-like)
    if (len(groups) >= 1 and len(groups) > 1) or ("ufac" in sheet_name.lower() and len(groups) >= 1):
        for _, row in df_raw.iterrows():
            name = None
            date = None
            # find name and date columns if present
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

    # LONG format detection: try to detect explicit columns
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

    # Fallback: treat row[0] as name, row[1] as date, remaining cells are jobids
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
    """
    More robust completion detection:
    - Check CaseCompletion column for 'completed' (not 'not completed' or 'incomplete')
    - If not present, check Comments
    """
    s = row.get("CaseCompletion")
    if s is not None and not pd.isna(s):
        s_lower = str(s).strip().lower()
        if "completed" in s_lower:
            return not ("not" in s_lower or "incomplete" in s_lower)
    c = row.get("Comments")
    if c is not None and not pd.isna(c):
        c_lower = str(c).strip().lower()
        if "completed" in c_lower:
            return not ("not" in c_lower or "incomplete" in c_lower)
    # fallback: if neither column indicates completion, return False
    return False

@st.cache_data
def load_and_process_data(uploaded_file):
    """
    Load Excel file with multiple sheets. Uses parse_sheet_generic to handle UFAC wide and AA long.
    Returns a processed DataFrame with columns:
      Source, Name, Date (datetime), JobID (string), CaseCompletion, Comments
    """
    try:
        xls = pd.ExcelFile(uploaded_file)
        all_rows = []
        # iterate sheets
        for sheet in xls.sheet_names:
            try:
                df_raw = pd.read_excel(uploaded_file, sheet_name=sheet, dtype=str, header=0)
            except Exception:
                df_raw = pd.read_excel(uploaded_file, sheet_name=sheet, dtype=str, header=None)
            parsed = parse_sheet_generic(df_raw, sheet)
            all_rows.extend(parsed)

        df = pd.DataFrame(all_rows, columns=["Source","Name","Date","JobID","CaseCompletion","Comments"])
        # coerce date
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        # normalize strings
        df["Source"] = df["Source"].astype(str).str.strip()
        df["Name"] = df["Name"].astype(str).str.strip()
        df["JobID"] = df["JobID"].astype(str).str.strip()

        # drop blank / obvious header rows
        df = df[~df["JobID"].str.lower().isin({"jobid","casecompletion","completed","comments","none","nan","job id"})]
        df = df[~df["Name"].astype(str).str.lower().isin({"name","nan","none"})]
        df = df.dropna(subset=["JobID"]).reset_index(drop=True)

        # completion marker
        df["IsCompleted"] = df.apply(detect_completed_marker, axis=1)

        return df
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# ---------------- AA DAILY CAP ----------------
def apply_aa_daily_cap(df, cap=12, apply_cap=True):
    """
    For rows where source contains 'aa' (case-insensitive), only mark the first `cap`
    cases per (Source, Name, Date(date-only)) as CountIncluded=True. Others get False.
    For non-AA sources, CountIncluded=True.
    This keeps the underlying rows intact, but provides a boolean we use for weighted aggregations.
    """
    df = df.copy().reset_index(drop=True)
    df["DateOnly"] = df["Date"].dt.date
    df["CountIncluded"] = True

    if not apply_cap:
        return df

    # Identify AA-like sources
    mask_aa = df["Source"].astype(str).str.lower().str.contains("aa")
    if not mask_aa.any():
        return df

    # Sort for deterministic selection (you may change sorting rule)
    aa_df = df[mask_aa].sort_values(by=["Source","Name","DateOnly","Date","JobID"]).copy()
    # cumcount per Source/Name/DateOnly
    aa_df["cum"] = aa_df.groupby(["Source","Name","DateOnly"]).cumcount()
    aa_df["IncludeFlag"] = aa_df["cum"] < cap

    # Map IncludeFlag back to main df
    df.loc[aa_df.index, "CountIncluded"] = aa_df["IncludeFlag"].values

    # cleanup helper columns
    df = df.drop(columns=["DateOnly"], errors="ignore")
    return df

# ---------------- AGGREGATION HELPERS ----------------
def df_included(df):
    """Return the subset of rows flagged as CountIncluded==True (if column exists); otherwise return df."""
    if "CountIncluded" in df.columns:
        return df[df["CountIncluded"] == True]
    return df

# ---------------- CHARTS / METRICS ----------------
def create_performance_metrics(df):
    """Use CountIncluded and IsCompleted for metrics (so AA cap is respected)."""
    df_proc = df_included(df)
    col1, col2, col3, col4 = st.columns(4)

    total_tasks = int(df_proc.shape[0])
    total_completed = int(df_proc["IsCompleted"].sum())
    unique_jobs = int(df_proc["JobID"].nunique())
    duplicates = int(df_proc.shape[0] - unique_jobs)

    with col1:
        st.metric("📋 Total Tasks (counted)", f"{total_tasks:,}", f"{total_completed:,} completed")
    with col2:
        st.metric("🔢 Unique Jobs (counted)", f"{unique_jobs:,}", f"{duplicates:,} duplicates" if duplicates > 0 else "No duplicates")
    with col3:
        st.metric("👥 Active Agents", f"{df['Name'].nunique():,}", f"{df['Source'].nunique():,} teams")
    with col4:
        completion_rate = (total_completed / total_tasks * 100) if total_tasks > 0 else 0.0
        st.metric("✅ Completion Rate (counted)", f"{completion_rate:.1f}%", f"{total_completed:,} of {total_tasks:,}")

def create_agent_performance_chart(df, top_n=10):
    """Aggregate using CountIncluded/IsCompleted per agent; show top_n by counted tasks."""
    inc = df_included(df)
    tasks_per_agent = inc.groupby("Name").agg(
        TaskCount=("JobID", "count"),
        Completed=("IsCompleted", "sum")
    ).reset_index().sort_values("TaskCount", ascending=False)
    tasks_per_agent["CompletionRate"] = (tasks_per_agent["Completed"] / tasks_per_agent["TaskCount"]) * 100
    top_agents = tasks_per_agent.head(top_n)

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Task Volume', 'Completion Rate'))
    fig.add_trace(go.Bar(x=top_agents["TaskCount"], y=top_agents["Name"], orientation='h', text=top_agents["TaskCount"], textposition='outside', name="Tasks"), row=1, col=1)
    fig.add_trace(go.Bar(x=top_agents["CompletionRate"], y=top_agents["Name"], orientation='h', text=[f"{v:.1f}%" for v in top_agents["CompletionRate"]], textposition='outside', name="Completion %"), row=1, col=2)
    fig.update_layout(height=600, showlegend=False, title_text=f"Top {top_n} Agent Performance (counted)")
    return fig

def create_team_comparison(df):
    """Team-level comparison using counted rows"""
    inc = df_included(df)
    team_stats = inc.groupby("Source").agg(
        TotalTasks=("JobID", "count"),
        CompletedTasks=("IsCompleted", "sum"),
        UniqueAgents=("Name", "nunique")
    ).reset_index()
    # handle division by zero
    team_stats["CompletionRate"] = np.where(team_stats["TotalTasks"]>0, (team_stats["CompletedTasks"] / team_stats["TotalTasks"]) * 100, 0.0)

    # Use domain for pie (plotly requirement)
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type":"bar"}, {"type":"bar"}],
               [{"type":"bar"}, {"type":"domain"}]],
        subplot_titles=('Tasks by Team', 'Completion Rate by Team', 'Agents by Team', 'Task Distribution')
    )

    fig.add_trace(go.Bar(x=team_stats["Source"], y=team_stats["TotalTasks"], name="Total Tasks"), row=1, col=1)
    fig.add_trace(go.Bar(x=team_stats["Source"], y=team_stats["CompletionRate"], name="Completion Rate"), row=1, col=2)
    fig.add_trace(go.Bar(x=team_stats["Source"], y=team_stats["UniqueAgents"], name="Unique Agents"), row=2, col=1)
    fig.add_trace(go.Pie(labels=team_stats["Source"], values=team_stats["TotalTasks"], name="Task Distribution"), row=2, col=2)

    fig.update_layout(height=800, showlegend=False)
    return fig

def create_time_trends(df):
    """Daily trends using CountIncluded weighting"""
    if df["Date"].isna().all():
        st.warning("No valid dates found for trend analysis")
        return None
    inc = df_included(df).dropna(subset=["Date"])
    if inc.empty:
        st.warning("No counted rows with valid dates for trend analysis")
        return None

    inc["DateOnly"] = inc["Date"].dt.date
    daily_stats = inc.groupby(["DateOnly","Source"]).agg(
        DailyTasks=("JobID","count"),
        DailyCompleted=("IsCompleted","sum")
    ).reset_index()
    daily_stats["CompletionRate"] = np.where(daily_stats["DailyTasks"]>0, (daily_stats["DailyCompleted"]/daily_stats["DailyTasks"])*100, 0.0)

    fig = make_subplots(rows=2, cols=1, subplot_titles=('Daily Task Volume by Team', 'Daily Completion Rate by Team'), shared_xaxes=True)
    for source in daily_stats["Source"].unique():
        sdata = daily_stats[daily_stats["Source"]==source]
        fig.add_trace(go.Scatter(x=sdata["DateOnly"], y=sdata["DailyTasks"], mode='lines+markers', name=f'{source} - Tasks'), row=1, col=1)
        fig.add_trace(go.Scatter(x=sdata["DateOnly"], y=sdata["CompletionRate"], mode='lines+markers', name=f'{source} - Completion %'), row=2, col=1)
    fig.update_layout(height=700)
    return fig

def create_duplicate_analysis(df):
    """Duplicates computed on counted rows (so duplicates that were excluded by AA cap won't appear)"""
    inc = df_included(df)
    dup_counts = inc.groupby("JobID").size().reset_index(name="Count")
    duplicates = dup_counts[dup_counts["Count"] > 1].sort_values("Count", ascending=False)
    if duplicates.empty:
        return None
    top_dups = duplicates.head(10)
    fig = px.bar(top_dups, x="Count", y="JobID", orientation='h', title="Top 10 Most Duplicated JobIDs", color="Count", color_continuous_scale="Reds")
    fig.update_layout(height=400)
    return fig, duplicates

# ---------------- TEAM INSIGHTS / TOP-MID-LOW ----------------
def create_team_overview(df):
    inc = df_included(df)
    team_stats = inc.groupby("Source").agg(TotalTasks=("JobID","count"), Completed=("IsCompleted","sum")).reset_index()
    team_stats["CompletionRate"] = np.where(team_stats["TotalTasks"]>0, (team_stats["Completed"]/team_stats["TotalTasks"])*100, 0.0)
    team_stats = team_stats.sort_values("CompletionRate", ascending=False)
    fig = px.bar(team_stats, x="CompletionRate", y="Source", orientation="h", title="Team Completion Rate (counted, sorted)", text="CompletionRate")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(height=450, xaxis_title="Completion Rate (%)")
    return fig, team_stats

def create_team_top_performers(df, top_n=5):
    inc = df_included(df)
    team_groups = inc.groupby(["Source","Name"]).agg(TaskCount=("JobID","count"), Completed=("IsCompleted","sum")).reset_index()
    team_groups["CompletionRate"] = np.where(team_groups["TaskCount"]>0, (team_groups["Completed"]/team_groups["TaskCount"])*100, 0.0)
    top_performers = team_groups.sort_values(["Source","TaskCount"], ascending=[True,False]).groupby("Source").head(top_n)
    if top_performers.empty:
        return None
    fig = px.bar(top_performers, x="TaskCount", y="Name", color="CompletionRate", facet_col="Source", orientation="h", text="CompletionRate", title=f"Top {top_n} Performers per Team (counted)")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(height=500, showlegend=False)
    return fig

def create_team_contribution_pie(df):
    inc = df_included(df)
    team_totals = inc.groupby("Source").agg(TotalTasks=("JobID","count")).reset_index()
    fig = px.pie(team_totals, names="Source", values="TotalTasks", title="Team Share of Total Tasks (counted)")
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    return fig

def create_team_trends_small_multiples(df):
    inc = df_included(df).dropna(subset=["Date"])
    if inc.empty:
        return None
    inc["DateOnly"] = inc["Date"].dt.date
    daily_stats = inc.groupby(["DateOnly","Source"]).agg(DailyTasks=("JobID","count"), DailyCompleted=("IsCompleted","sum")).reset_index()
    daily_stats["CompletionRate"] = np.where(daily_stats["DailyTasks"]>0, (daily_stats["DailyCompleted"]/daily_stats["DailyTasks"])*100, 0.0)
    fig = px.line(daily_stats, x="DateOnly", y="DailyTasks", color="Source", facet_col="Source", facet_col_wrap=3, title="Daily Task Volume (small multiples by Team)")
    fig.update_layout(height=600, showlegend=False)
    return fig

def create_duplicates_heatmap(df, top_n_jobids=20):
    inc = df_included(df)
    dup_counts = inc.groupby("JobID").size().reset_index(name="Count")
    dup_jobids = dup_counts[dup_counts["Count"] > 1]["JobID"].head(top_n_jobids)
    if dup_jobids.empty:
        return None
    pivot = inc[inc["JobID"].isin(dup_jobids)].assign(flag=1).pivot_table(index="JobID", columns="Source", values="flag", aggfunc="sum", fill_value=0)
    if pivot.empty:
        return None
    fig = px.imshow(pivot, text_auto=True, aspect="auto", color_continuous_scale="Reds", title="🔍 Duplicate JobID Heatmap (Top Duplicates across Teams)")
    return fig

# ---------------- TOP / MID / LOW FUNCTION (using counted rows) ----------------
@st.cache_data
def split_agents_by_performance(df, freq="W", rank_by="Tasks"):
    inc = df_included(df).dropna(subset=["Date"])
    if inc.empty:
        return []
    # aggregate per period/team/agent
    agg = (
        inc.groupby([pd.Grouper(key="Date", freq=freq), "Source", "Name"])
        .agg(Tasks=("JobID","count"), Completed=("IsCompleted","sum"))
        .reset_index()
    )
    agg["CompletionRate"] = np.where(agg["Tasks"]>0, (agg["Completed"]/agg["Tasks"])*100, 0.0)

    results = []
    for (period, team), group in agg.groupby(["Date","Source"]):
        g = group.copy().reset_index(drop=True)
        if rank_by == "Tasks":
            g = g.sort_values("Tasks", ascending=False).reset_index(drop=True)
        else:
            g = g.sort_values(["CompletionRate","Tasks"], ascending=[False, False]).reset_index(drop=True)
        n = len(g)
        if n == 0:
            continue
        # split into top/mid/low (ceil)
        top_end = max(1, int(np.ceil(n/3)))
        mid_end = int(np.ceil(2*n/3))
        top_df = g.iloc[:top_end].reset_index(drop=True)
        mid_df = g.iloc[top_end:mid_end].reset_index(drop=True)
        low_df = g.iloc[mid_end:].reset_index(drop=True)
        for dfb in (top_df, mid_df, low_df):
            if not dfb.empty:
                dfb["PeriodStart"] = period
                dfb["Team"] = team
        results.append({"Period":period, "Team":team, "Top":top_df, "Mid":mid_df, "Low":low_df})
    return results

def display_agent_split_tables(splits, label=""):
    if not splits:
        st.info(f"No data for {label} Top/Mid/Low.")
        return
    st.markdown(f"### {label} Top / Mid / Low (by Team & Period)")
    # Sort by Team then Period for stable ordering
    for s in sorted(splits, key=lambda x: (x["Team"], pd.to_datetime(x["Period"]))):
        period = s["Period"]
        try:
            period_str = pd.to_datetime(period).strftime("%Y-%m-%d")
        except Exception:
            period_str = str(period)
        with st.expander(f"{s['Team']} — period starting {period_str}"):
            st.write("Top performers")
            if not s["Top"].empty:
                df_top = s["Top"][["Name","Tasks","Completed","CompletionRate"]].copy().rename(columns={"Tasks":"Cases","Completed":"Completed","CompletionRate":"% Completed"})
                df_top["% Completed"] = df_top["% Completed"].round(1)
                st.dataframe(df_top.reset_index(drop=True))
            else:
                st.write("No Top performers for this period.")
            st.write("Mid performers")
            if not s["Mid"].empty:
                df_mid = s["Mid"][["Name","Tasks","Completed","CompletionRate"]].copy().rename(columns={"Tasks":"Cases","Completed":"Completed","CompletionRate":"% Completed"})
                df_mid["% Completed"] = df_mid["% Completed"].round(1)
                st.dataframe(df_mid.reset_index(drop=True))
            else:
                st.write("No Mid performers for this period.")
            st.write("Low performers")
            if not s["Low"].empty:
                df_low = s["Low"][["Name","Tasks","Completed","CompletionRate"]].copy().rename(columns={"Tasks":"Cases","Completed":"Completed","CompletionRate":"% Completed"})
                df_low["% Completed"] = df_low["% Completed"].round(1)
                st.dataframe(df_low.reset_index(drop=True))
            else:
                st.write("No Low performers for this period.")

# ---------------- MAIN ----------------
def main():
    st.markdown('<div class="main-header">📊 GTM Team Performance Dashboard</div>', unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx','xls'])
        st.markdown("---")
        st.header("🔍 Filters")
        top_n = st.slider("Top N Agents to Show", 3, 30, 10)
        teams_placeholder = st.empty()  # we'll fill after loading data
        st.markdown("---")
        period_choice = st.selectbox("Period granularity", ["Weekly","Monthly","Yearly"], index=0)
        rank_by = st.radio("Rank Top/Mid/Low by", ["Tasks","CompletionRate"], index=0)
        show_topmidlow = st.checkbox("Show Top / Mid / Low section", value=True)
        st.markdown("---")
        st.subheader("AA Cap")
        apply_aa_cap = st.checkbox("Apply GTM AA daily cap (max 12 cases/day)", value=True)
        aa_cap_value = st.number_input("AA daily cap (cases per agent/day)", min_value=1, max_value=100, value=12, step=1)

    if uploaded_file is None:
        st.info("👆 Please upload an Excel file to begin analysis")
        return

    # Load & parse
    with st.spinner("Processing uploaded file..."):
        df = load_and_process_data(uploaded_file)

    if df is None or df.empty:
        st.warning("No valid records found in the uploaded file.")
        return

    # Apply AA cap
    df = apply_aa_daily_cap(df, cap=int(aa_cap_value), apply_cap=apply_aa_cap)

    # Team selector (populate now that df exists)
    teams = ["All"] + sorted(df["Source"].dropna().unique().tolist())
    selected_team = st.sidebar.selectbox("Select Team", teams)

    # Date range filter
    if not df["Date"].isna().all():
        min_date = df["Date"].min().date()
        max_date = df["Date"].max().date()
        date_range = st.sidebar.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    else:
        date_range = None

    # apply filters
    filtered = df.copy()
    if selected_team != "All":
        filtered = filtered[filtered["Source"] == selected_team]
    if date_range and len(date_range) == 2:
        filtered = filtered[(filtered["Date"].dt.date >= date_range[0]) & (filtered["Date"].dt.date <= date_range[1])]

    if filtered.empty:
        st.warning("No data matches the current filters.")
        return

    # MAIN DASHBOARD
    st.header("📊 Key Metrics")
    create_performance_metrics(filtered)

    st.header("👤 Agent Performance")
    try:
        st.plotly_chart(create_agent_performance_chart(filtered, top_n=top_n), use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting agent performance: {e}")

    # Team comparison (only for All or show team if chosen)
    if selected_team == "All":
        st.header("🏆 Team Comparison")
        try:
            st.plotly_chart(create_team_comparison(filtered), use_container_width=True)
        except Exception as e:
            st.error(f"Error plotting team comparison: {e}")

        st.subheader("Team Completion Rates")
        try:
            overview_fig, team_stats = create_team_overview(filtered)
            st.plotly_chart(overview_fig, use_container_width=True)
            with st.expander("View Team Stats Table"):
                st.dataframe(team_stats)
        except Exception as e:
            st.error(f"Error creating team overview: {e}")

        st.subheader("🌟 Top Performers per Team")
        try:
            top_perf_fig = create_team_top_performers(filtered, top_n=top_n if top_n <= 20 else 20)
            if top_perf_fig is not None:
                st.plotly_chart(top_perf_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating top performers: {e}")

        st.subheader("📌 Team Contribution")
        try:
            st.plotly_chart(create_team_contribution_pie(filtered), use_container_width=True)
        except Exception as e:
            st.error(f"Error creating team contribution pie: {e}")

        st.subheader("📈 Team Trends (small multiples)")
        small_trends = create_team_trends_small_multiples(filtered)
        if small_trends:
            st.plotly_chart(small_trends, use_container_width=True)

        st.subheader("🔍 Duplicate JobID Heatmap")
        dup_heat = create_duplicates_heatmap(filtered, top_n_jobids=20)
        if dup_heat:
            st.plotly_chart(dup_heat, use_container_width=True)
        else:
            st.info("No duplicated JobIDs to display in heatmap.")
    else:
        st.subheader(f"Team: {selected_team} — Breakdown")
        # team-specific insights
        try:
            single_team_top = create_team_top_performers(filtered, top_n=top_n)
            if single_team_top:
                st.plotly_chart(single_team_top, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating team top performers: {e}")

        st.subheader("Team Trends Over Time")
        try:
            team_trend_fig = create_time_trends(filtered)
            if team_trend_fig:
                st.plotly_chart(team_trend_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating time trends: {e}")

        st.subheader("Duplicate Analysis (Team)")
        try:
            dup_result = create_duplicate_analysis(filtered)
            if dup_result:
                dup_chart, dup_data = dup_result
                st.plotly_chart(dup_chart, use_container_width=True)
                with st.expander("View Duplicate Details"):
                    st.dataframe(dup_data)
            else:
                st.info("No duplicates for this team.")
        except Exception as e:
            st.error(f"Error creating duplicate analysis: {e}")

    # Top / Mid / Low
    if show_topmidlow:
        st.markdown("---")
        st.header("📊 Top / Mid / Low Performers by Period (per Team)")
        freq_map = {"Weekly":"W", "Monthly":"M", "Yearly":"Y"}
        freq = freq_map.get(period_choice, "W")
        splits = split_agents_by_performance(filtered, freq=freq, rank_by=rank_by)
        display_agent_split_tables(splits, label=period_choice)

    # Raw Data
    with st.expander("🔍 View Raw Data (first 500 rows)"):
        st.dataframe(filtered.head(500), use_container_width=True)

if __name__ == "__main__":
    main()
