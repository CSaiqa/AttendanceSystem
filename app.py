import os
import streamlit as st
import mysql.connector
import pandas as pd

# Set up MySQL connection
conn = mysql.connector.connect(
    host="localhost",
    user="Cse299",
    password="12345",
    database="attendance_system"
)
cursor = conn.cursor()

# Function to fetch attendance records
def fetch_attendance():
    query = """
    SELECT a.student_id, s.name, a.timestamp
    FROM Attendance a
    JOIN Students s ON a.student_id = s.student_id
    """
    cursor.execute(query)
    records = cursor.fetchall()
    return [{"Student ID": r[0], "Name": r[1], "Timestamp": r[2]} for r in records]

# Streamlit app interface
st.title("Facial Recognition Attendance System")

# Button to run `atten2.py`
if st.button("Start Recognition"):
    st.info("Running facial recognition system...")
    try:
        os.system("python atten2.py")  # This runs the atten2.py script
        st.success("Facial recognition system is running.")
    except Exception as e:
        st.error(f"Error while running recognition: {e}")

# Button to display attendance table
if st.button("Display Attendance Table"):
    records = fetch_attendance()
    if records:
        df = pd.DataFrame(records)
        st.table(df)
    else:
        st.write("No attendance records found.")

# Close the MySQL connection
conn.close()