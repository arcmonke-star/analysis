# Full updated Colab-ready analysis script
# - Correctly parses GTM UFAC (wide repeating JobID/Case/Comments)
# - Parses GTM AA (long/wide) robustly
# - Normalizes JobIDs, removes header artifacts
# - Produces team + agent analyses (duplicates, completion, reliability, trends)
# - Colab: mounts Drive, reads the first Excel in folder_path

import os, re
from decimal import Decimal, InvalidOperation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
from IPython.display import display

# ---------------- PARAMETERS ----------------
top_n = 10
data_folder = "/content/drive/MyDrive/Colab Notebooks/Workflow"

# ---------------- SETUP ----------------
drive.mount('/content/drive', force_remount=False)
sns.set(style="whitegrid")
plt.rcParams.update({'figure.autolayout': True})

# ---------------- HELPERS ----------------
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

def safe_show():
    try:
        plt.show()
    except Exception:
        pass

def is_header_like_name(x):
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return s in {"name","jobid","casecompletion","comments","completed","none","nan","job id"}

# ---------------- LOCATE EXCEL ----------------
if not os.path.exists(data_folder):
    raise FileNotFoundError(f"Folder not found: {data_folder}")

xls_files = [f for f in os.listdir(data_folder) if f.lower().endswith(('.xlsx', '.xls'))]
if not xls_files:
    raise FileNotFoundError(f"No Excel files found in {data_folder}")

file_path = os.path.join(data_folder, xls_files[0])
print("Using file:", file_path)

# ---------------- PARSERS ----------------
def find_repeating_groups(cols_lower):
    """
    Identify positions of repeating groups like [job*, case*, comment*] in the column list
    Returns list of tuples (job_idx, case_idx_or_None, comment_idx_or_None)
    """
    groups = []
    i = 0
    n = len(cols_lower)
    while i < n:
        col = cols_lower[i]
        if "job" in col:  # job column detected
            j = i
            # look ahead for a 'case' column
            case_idx = None
            comment_idx = None
            # find nearest case column after j (within next 6 cols)
            for k in range(i+1, min(n, i+6)):
                if "case" in cols_lower[k] or "completion" in cols_lower[k]:
                    case_idx = k
                    # find comment after case
                    for m in range(k+1, min(n, k+6)):
                        if "comment" in cols_lower[m]:
                            comment_idx = m
                            break
                    break
            groups.append((j, case_idx, comment_idx))
            # advance i — if case exists, jump after it, else just next
            if case_idx:
                i = case_idx + 1
            else:
                i = j + 1
        else:
            i += 1
    return groups

def parse_sheet_generic(df_raw, sheet_name):
    """
    Universal sheet parser:
    - Detect repeating job groups -> treat as wide and unpivot
    - Else try to find Name/Date/JobID/CaseCompletion columns (long format)
    """
    rows = []
    # normalize column names for searching
    cols = list(df_raw.columns)
    cols_lower = [str(c).strip().lower() for c in cols]

    # detect repeating groups
    groups = find_repeating_groups(cols_lower)

    # If many job groups (more than 1), treat as wide repeating layout (UFAC)
    if len(groups) >= 1 and len(groups) > 1 or ("ufac" in sheet_name.lower() and len(groups) >= 1):
        # For each row, iterate all groups
        for idx, row in df_raw.iterrows():
            # Attempt to find name and date from obvious columns if present
            name = None
            date = None
            # heuristics to find name/date columns: first column containing 'name' or first col
            for c, low in zip(cols, cols_lower):
                if "name" in low:
                    name = row.get(c)
                    break
            if name is None:
                # fallback: first column value
                name = row.iloc[0] if df_raw.shape[1] > 0 else None
            for c, low in zip(cols, cols_lower):
                if "date" in low:
                    date = row.get(c)
                    break
            if date is None and df_raw.shape[1] > 1:
                date = row.iloc[1]  # common layout: name, date, then jobs

            # For each group extract job/case/comment
            for (jidx, cidx, commidx) in groups:
                try:
                    job_col = cols[jidx]
                    job_val = row.get(job_col)
                except Exception:
                    job_val = None
                if pd.isna(job_val) or str(job_val).strip() == "":
                    continue
                # case
                case_val = None
                if cidx is not None:
                    case_col = cols[cidx]
                    case_val = row.get(case_col)
                # comment
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

    # Else - try long format: find Name, Date, JobID, CaseCompletion columns by header keywords
    # Map best-match columns
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

    # If job_col found, treat as long: each row -> one job
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

    # As fallback, try headerless / first-col-name layout (name, date, then many job ids)
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

# ---------------- READ ALL SHEETS ----------------
xls = pd.ExcelFile(file_path)
all_rows = []
for sheet in xls.sheet_names:
    print("Processing sheet:", sheet)
    # read with header=0 - we need raw headers to detect repeating patterns
    try:
        df_raw = pd.read_excel(file_path, sheet_name=sheet, dtype=str, header=0)
    except Exception as e:
        # fallback try header=None
        df_raw = pd.read_excel(file_path, sheet_name=sheet, dtype=str, header=None)
    parsed = parse_sheet_generic(df_raw, sheet)
    all_rows.extend(parsed)

# ---------------- BUILD DF & CLEAN ----------------
df = pd.DataFrame(all_rows, columns=["Source","Name","Date","JobID","CaseCompletion","Comments"])
# normalize
df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
df["JobID"] = df["JobID"].astype(str).str.strip()
# drop header-like rows and empty job ids
df = df[~df["JobID"].str.lower().isin({"jobid","casecompletion","completed","comments","none","nan","job id"})]
df = df[~df["Name"].astype(str).str.lower().isin({"name","nan","none"})]
df = df.dropna(subset=["JobID"]).reset_index(drop=True)

print("\nSample cleaned data (first 12 rows):")
display(df.head(12))

# ---------------- DERIVED FIELDS ----------------
# Robust hybrid detection that handles both AA and UFAC varieties
def detect_completed_marker(row):
    source = row["Source"]

    # ------------------- GTM AA -------------------
    if "aa" in source.lower():
        s = row["CaseCompletion"]
        if s is None or pd.isna(s):
            return False
        s_lower = str(s).strip().lower()
        if "completed" in s_lower:
            return "not" not in s_lower and "incomplete" not in s_lower
        return False

    # ------------------- GTM UFAC -------------------
    if "ufac" in source.lower():
        jobid = str(row["JobID"]).strip().lower()
        # If JobID is purely numeric, it's just an ID → not a completion marker
        if jobid.isnumeric():
            return False
        # Otherwise, it's an outcome string → count as completed
        return True

    # ------------------- Default -------------------
    return False

df["IsCompleted"] = df.apply(detect_completed_marker, axis=1)

# ---------------- METRICS ----------------
# Basic counts
total_rows = len(df)
unique_jobs = df["JobID"].nunique()
unique_agents = df["Name"].nunique()
date_min = df["Date"].min()
date_max = df["Date"].max()
print(f"\nRows: {total_rows}, Unique Jobs: {unique_jobs}, Agents: {unique_agents}, Date range: {date_min} -> {date_max}")

# Tasks per agent
tasks_per_agent = df.groupby("Name").agg(
    TaskCount=("JobID","count"),
    Completed=("IsCompleted","sum")
).reset_index().sort_values("TaskCount", ascending=False)
tasks_per_agent["CompletionRate"] = (tasks_per_agent["Completed"] / tasks_per_agent["TaskCount"]) * 100

# Duplicate JobIDs summary
dup_counts = df.groupby("JobID").size().reset_index(name="Count")
dup_counts = dup_counts[dup_counts["Count"] > 1].sort_values("Count", ascending=False)

# Team-level stats
team_stats = df.groupby("Source").agg(TotalTasks=("JobID","count"), CompletedTasks=("IsCompleted","sum")).reset_index()
team_stats["CompletionRate"] = (team_stats["CompletedTasks"] / team_stats["TotalTasks"]) * 100
print("\nTeam stats:")
display(team_stats)

# ---------------- PLOTS & ANALYSES ----------------
# Top agents by volume
plt.figure(figsize=(10,6))
sns.barplot(data=tasks_per_agent.head(top_n), x="TaskCount", y="Name", palette="viridis")
plt.title(f"Top {top_n} Agents by Task Volume")
plt.tight_layout(); safe_show()

# Completed counts for top agents
plt.figure(figsize=(10,6))
sns.barplot(data=tasks_per_agent.head(top_n), x="Completed", y="Name", palette="magma")
plt.title(f"Completed Cases by Top {top_n} Agents")
plt.tight_layout(); safe_show()

# Duplicates
if not dup_counts.empty:
    topdup = dup_counts.head(10)
    print("\nTop duplicated JobIDs (JobID, Count):")
    display(topdup)
    plt.figure(figsize=(10,6))
    sns.barplot(data=topdup, x="Count", y="JobID", palette="coolwarm")
    plt.title("Top 10 Duplicate JobIDs")
    plt.tight_layout(); safe_show()
else:
    print("\nNo duplicate JobIDs found.")

# Team comparison bar
plt.figure(figsize=(8,6))
sns.barplot(data=team_stats, x="TotalTasks", y="Source", palette="Set2")
plt.title("Total Tasks per Team")
plt.tight_layout(); safe_show()

# Performer segmentation per team
tasks_per_agent_team = df.groupby(["Source","Name"]).agg(TaskCount=("JobID","count"), Completed=("IsCompleted","sum")).reset_index()
def rank_performers_local(team_df):
    team_df = team_df.sort_values("TaskCount", ascending=False).reset_index(drop=True)
    n = len(team_df)
    if n == 0:
        return team_df
    team_df["RankPercentile"] = (team_df.index + 1) / n * 100
    team_df["Category"] = pd.cut(team_df["RankPercentile"], bins=[0,20,80,100], labels=["Top Performer","Mid Performer","Low Performer"])
    return team_df

ranked_performers = tasks_per_agent_team.groupby("Source", group_keys=False).apply(rank_performers_local)
print("\nRanked performers (sample):")
display(ranked_performers.head(40))

plt.figure(figsize=(10,6))
sns.countplot(data=ranked_performers, x="Source", hue="Category", palette="coolwarm")
plt.title("Performer Distribution per Team")
plt.tight_layout(); safe_show()

# Team trends
tasks_per_day_team = df.groupby(["Date","Source"])["JobID"].count().reset_index(name="DailyTasks")
if not tasks_per_day_team.empty:
    plt.figure(figsize=(12,6))
    sns.lineplot(data=tasks_per_day_team, x="Date", y="DailyTasks", hue="Source", marker="o")
    plt.title("Daily Tasks Trend per Team")
    plt.xticks(rotation=45)
    plt.tight_layout(); safe_show()
else:
    print("No date-based trend data.")

# Advanced: agent consistency (boxplot of daily counts)
daily_tasks = df.groupby(["Name","Date"])["JobID"].count().reset_index(name="DailyCount")
if not daily_tasks.empty:
    plt.figure(figsize=(14,6))
    # show only top N agents to avoid overcrowding; else top contributors
    top_agents_for_box = tasks_per_agent.head(30)["Name"].tolist()
    box_df = daily_tasks[daily_tasks["Name"].isin(top_agents_for_box)]
    sns.boxplot(data=box_df, x="Name", y="DailyCount")
    plt.xticks(rotation=90)
    plt.title("Agent daily task distribution (Top agents shown)")
    plt.ylabel("Tasks per Day")
    plt.tight_layout(); safe_show()

# Duplicate burden by agent
dup_agents = df[df["JobID"].isin(dup_counts["JobID"])].groupby("Name")["JobID"].count().reset_index(name="DupCount")
if not dup_agents.empty:
    dup_agents = dup_agents.sort_values("DupCount", ascending=False)
    plt.figure(figsize=(10,6))
    sns.barplot(data=dup_agents.head(top_n), x="DupCount", y="Name", palette="Reds")
    plt.title("Top Agents by Duplicate JobIDs")
    plt.tight_layout(); safe_show()
else:
    print("No duplicate burden by agent.")

# Outlier scatter (tasks vs completion rate)
tasks_per_agent["CompletionRate"] = (tasks_per_agent["Completed"] / tasks_per_agent["TaskCount"]) * 100
plt.figure(figsize=(10,6))
sns.scatterplot(data=tasks_per_agent, x="TaskCount", y="CompletionRate", s=100)
plt.title("Tasks vs Completion Rate (agents)")
plt.xlabel("Total Tasks")
plt.ylabel("Completion Rate (%)")
plt.tight_layout(); safe_show()

# Reliability score per agent
# Need per-agent DailyStd and DupCount
daily_std = daily_tasks.groupby("Name")["DailyCount"].std().rename("DailyStd")
dup_count_agent = dup_agents.set_index("Name")["DupCount"] if not dup_agents.empty else pd.Series(dtype=int)

agent_metrics = tasks_per_agent.set_index("Name").join(daily_std).fillna(0)
agent_metrics["DailyStd"] = agent_metrics["DailyStd"].fillna(0)
agent_metrics["DupCount"] = agent_metrics.index.map(lambda n: int(dup_count_agent.get(n, 0)))
# Avoid division by zero in normalization
max_std = agent_metrics["DailyStd"].max() if agent_metrics["DailyStd"].max() > 0 else 1
max_dup = agent_metrics["DupCount"].max() if agent_metrics["DupCount"].max() > 0 else 1

agent_metrics["ReliabilityScore"] = (
    (agent_metrics["CompletionRate"]/100) * 0.5 +
    (1 - (agent_metrics["DailyStd"]/max_std)) * 0.3 +
    (1 - (agent_metrics["DupCount"]/max_dup)) * 0.2
) * 100

top_reliable = agent_metrics.sort_values("ReliabilityScore", ascending=False).head(top_n)
print("\nTop reliable agents (sample):")
display(top_reliable[["TaskCount","Completed","CompletionRate","DailyStd","DupCount","ReliabilityScore"]])

plt.figure(figsize=(10,6))
sns.barplot(data=top_reliable.reset_index(), x="ReliabilityScore", y="Name", palette="Blues")
plt.title("Top Agents With Less Duplicate Cases (composite score)")
plt.tight_layout(); safe_show()

# ---------------- UFAC team deep-dive ----------------
ufac_df = df[df["Source"].str.contains("ufac", case=False)]
if not ufac_df.empty:
    print("\nUFAC-specific metrics:")
    ufac_tasks = ufac_df.groupby("Name").agg(TaskCount=("JobID","count"), Completed=("IsCompleted","sum")).reset_index().sort_values("TaskCount", ascending=False)
    ufac_tasks["CompletionRate"] = (ufac_tasks["Completed"]/ufac_tasks["TaskCount"])*100
    display(ufac_tasks.head(20))

    plt.figure(figsize=(10,6))
    sns.barplot(data=ufac_tasks.head(top_n), x="TaskCount", y="Name", palette="viridis")
    plt.title(f"Top {top_n} UFAC Agents by Task Volume")
    plt.tight_layout(); safe_show()

    plt.figure(figsize=(10,6))
    sns.barplot(data=ufac_tasks.sort_values("CompletionRate", ascending=False).head(30), x="CompletionRate", y="Name", palette="magma")
    plt.title("UFAC Agents: Completion Rate (%)")
    plt.tight_layout(); safe_show()
else:
    print("No UFAC data found.")

# ---------------- AA team deep-dive ----------------
aa_df = df[df["Source"].str.contains("aa", case=False)]
if not aa_df.empty:
    print("\nAA-specific metrics:")
    aa_tasks = aa_df.groupby("Name").agg(TaskCount=("JobID","count"), Completed=("IsCompleted","sum")).reset_index().sort_values("TaskCount", ascending=False)
    aa_tasks["CompletionRate"] = (aa_tasks["Completed"]/aa_tasks["TaskCount"])*100
    display(aa_tasks.head(60))

    plt.figure(figsize=(10,6))
    sns.barplot(data=aa_tasks.head(top_n), x="TaskCount", y="Name", palette="viridis")
    plt.title(f"Top {top_n} AA Agents by Task Volume")
    plt.tight_layout(); safe_show()

    plt.figure(figsize=(10,6))
    sns.barplot(data=aa_tasks.sort_values("CompletionRate", ascending=False).head(30), x="CompletionRate", y="Name", palette="magma")
    plt.title("AA Agents: Completion Rate (%)")
    plt.tight_layout(); safe_show()
else:
    print("No AA data found.")

print("\nAnalysis finished. If some charts look crowded, reduce top_n or limit agent lists for boxplots.")
