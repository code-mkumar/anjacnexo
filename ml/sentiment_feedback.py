import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import os

dataset = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dbs/college_dataset.csv"))
# Load dataset from CSV
df = pd.read_csv(dataset)

# Train the machine learning model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(df["query"], df["prediction"])
query_db = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dbs/query_responses.db"))
# Connect to SQLite database
conn = sqlite3.connect(query_db)
cursor = conn.cursor()
def create_connection():
    return sqlite3.connect(query_db)
# Create history table if not exists
# cursor.execute("""
# CREATE TABLE IF NOT EXISTS history (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     query TEXT,
#     response TEXT
# )
# """)
cursor.execute("""
CREATE TABLE IF NOT EXISTS history1 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT,
    response TEXT,
    message TEXT
)
""")
cursor.execute("PRAGMA table_info(history1);")
columns = cursor.fetchall()

# Print table structure
# print("Table Structure of history1:")
# for column in columns:
#     print(column)

# Close the connection
# conn.close()

conn.commit()

# Function to retrieve the last query-response from the database
def get_last_query_from_session(dlist=[]):
    #cursor.execute("SELECT query, response FROM history ORDER BY id DESC LIMIT 1")
    last_qa = dlist[-1] 
    return last_qa  # Returns (query, response) or None if empty

# Function to predict and store results in the database
def predict_and_store(query,dlist):
    prediction = model.predict([query])[0]

    if prediction == "yes":
        # Fetch last stored query-response from the database
        last_entry = get_last_query_from_session(dlist)
        if last_entry:
            last_query=last_entry["question"]
            last_response=last_entry["answer"]
            print(last_entry,last_response)
            cursor.execute("INSERT INTO history1 (query, response,message) VALUES (?, ?,?);", (last_query, last_response,query))
            conn.commit()
            conn.close()
        return f"this is feedback from the user"
    return " "
    
    # else:
    #     # Store the current query-response in the database
    #     #cursor.execute("INSERT INTO history (query, response) VALUES (?, ?)", (query, prediction))
    #     conn.commit()
    #     return prediction

# Interactive testing
# while True:
#     user_query = input("Enter a query (or type 'exit' to stop): ")
#     if user_query.lower() == "exit":
#         break
#     result = predict_and_store(user_query)
#     print(f"Prediction: {result}")

# #Fetch and display stored history
# cursor.execute("SELECT * FROM history1")
# res = cursor.fetchall()
# for i in res:
#     print(i)
# # cursor.execute("delete from history")
# # conn.commit()
# print("hai")
# cursor.execute("SELECT * FROM history")
# res = cursor.fetchall()
# for i in res:
#     print(i)
# # Close the database connection
# conn.close()

def view_table():
    conn = create_connection()
    cursor=conn.cursor()
    cursor.execute("""
    SELECT * FROM history1;""")
    rows = cursor.fetchall()
    return rows