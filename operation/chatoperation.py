import sqlite3
import os
from datetime import datetime

def create_connection():
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dbs/chat.db"))
    return sqlite3.connect(db_path)

def create_table():
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            session_name TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            relevant_chunks_idx TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()

# 1️⃣ AUTOMATE SESSION CREATION
def generate_session_name(user_id):
    """Automatically generates a new session name based on user history."""
    conn = create_connection()
    cursor = conn.cursor()
    
    # Count existing sessions for the user
    cursor.execute("""
        SELECT COUNT(DISTINCT session_name) FROM chats WHERE user_id = ?
    """, (user_id,))
    
    session_count = cursor.fetchone()[0] + 1  # Increment count
    session_name = f"_{user_id}_session_{session_count}"
    
    conn.close()
    return session_name

# 2️⃣ ADD CHAT ENTRY
def add_chat(user_id, question, answer, idx='', session_name=None):
    """Adds chat data. Creates a new session if not provided."""
    if session_name is None:
        session_name = generate_session_name(user_id)  # Auto-generate if missing

    conn = create_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO chats (user_id, session_name, question, answer, relevant_chunks_idx)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, session_name, question, answer, idx))
    
    conn.commit()
    conn.close()
    return session_name  # Return session name for tracking

# 3️⃣ GET CHAT HISTORY
def get_chat_history(user_id, session_name):
    """Fetches chat history for a specific session."""
    conn = create_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT question, answer, timestamp ,relevant_chunks_idx FROM chats
        WHERE user_id = ? AND session_name = ?
        ORDER BY timestamp ASC
    """, (user_id, session_name))
    
    history = cursor.fetchall()
    conn.close()
    return history

# 4️⃣ LIST USER SESSIONS
def get_user_sessions(user_id):
    """Gets all unique session names for a user."""
    conn = create_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT DISTINCT session_name FROM chats WHERE user_id = ?
    """, (user_id,))
    
    sessions = [row[0] for row in cursor.fetchall()]
    conn.close()
    return sessions

# 5️⃣ RENAME SESSION
def rename_session(user_id, old_name, new_name):
    """Renames a session for a user."""
    conn = create_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE chats SET session_name = ? 
        WHERE user_id = ? AND session_name = ?
    """, (new_name, user_id, old_name))
    
    conn.commit()
    conn.close()

# 6️⃣ DELETE SESSION
def delete_session(user_id, session_name):
    """Deletes an entire session for a user."""
    conn = create_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        DELETE FROM chats WHERE user_id = ? AND session_name = ?
    """, (user_id, session_name))
    
    conn.commit()
    conn.close()
create_table()
