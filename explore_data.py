# explore_data.py
# Exploratory Data Analysis for ContextRec

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Config ===
RAW_DATA_PATH = "./data/events.csv"
REPORTS_DIR = "./reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# === Load Data ===
print("Loading data...")
df = pd.read_csv(RAW_DATA_PATH)

# === Convert Timestamp ===
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df['date'] = df['timestamp'].dt.date

# === Event Type Distribution ===
plt.figure(figsize=(6,4))
df['event'].value_counts().plot(kind='bar', color='skyblue')
plt.title("Event Type Distribution")
plt.xlabel("Event Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/event_type_distribution.png")
plt.close()

# === Events Over Time ===
daily_events = df.groupby(['date', 'event']).size().unstack().fillna(0)
daily_events.plot(figsize=(12, 5))
plt.title("Daily Event Volume by Type")
plt.xlabel("Date")
plt.ylabel("Number of Events")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/events_over_time.png")
plt.close()


# ===  Session Length Distribution ===
sessions = df.groupby(['visitorid', 'date'])['itemid'].agg(list).reset_index()
sessions['session_length'] = sessions['itemid'].apply(len)
plt.figure(figsize=(8, 4))
sns.histplot(sessions['session_length'], bins=30, kde=True)
plt.title("Distribution of Session Lengths")
plt.xlabel("Items per Session")
plt.ylabel("Number of Sessions")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/session_lengths.png")
plt.close()

sessions['formatted_date'] = pd.to_datetime(sessions['date'])
session_length_by_day = (
    sessions.groupby("formatted_date")["session_length"]
    .mean()
    .reset_index()
)
plt.figure(figsize=(10, 4))
sns.lineplot(data=session_length_by_day, x="formatted_date", y="session_length")
plt.title("Average Session Length Over Time")
plt.xlabel("Date")
plt.ylabel("Avg. Items per Session")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/avg_session_length_over_time.png")
plt.close()



print("Reports saved in ./reports/")
