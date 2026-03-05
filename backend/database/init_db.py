import sqlite3

conn = sqlite3.connect("database/feedback.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    input_text TEXT,
    agents_used TEXT,
    output TEXT,
    rating INTEGER
)
""")

conn.commit()
conn.close()