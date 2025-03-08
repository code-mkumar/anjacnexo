import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Function to extract timestamp, question, and answer correctness
def extract_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    data = []
    current_question = None
    current_time = None

    for line in lines:
        if line.lower().startswith("question:"):
            current_question = line.split(":", 1)[1].strip()
        elif line.lower().startswith("time:"):
            current_time = line.split(":", 1)[1].strip().split()[0]  # Extract date only (YYYY-MM-DD)
        elif line.lower().startswith("answer:"):
            answer_text = line.split(":", 1)[1].strip().lower()
            correctness = "correct" if "error" not in answer_text and "not found" not in answer_text else "wrong"
            if current_question and current_time:
                data.append((current_time, current_question, correctness))
    
    return pd.DataFrame(data, columns=["date", "question", "correctness"])

# File path
file_path = "./dbs/data.txt"

# Extract data
df = extract_data(file_path)
print(len(df))
# Convert date to datetime and group by week
df["date"] = pd.to_datetime(df["date"])
df["week"] = df["date"].dt.strftime("%Y-%W")  # Format as Year-Week number

# Count correct and wrong responses per week
weekly_counts = df.groupby(["week", "correctness"]).size().unstack(fill_value=0)

# Plot the data
plt.figure(figsize=(12, 6))
weekly_counts.plot(kind="bar", stacked=True, color=["green", "red"])
plt.xlabel("Week")
plt.ylabel("Number of Responses")
plt.title("Correct vs. Wrong Responses Per Week")
plt.legend(["Correct", "Wrong"])
plt.xticks(rotation=45)
plt.show()
