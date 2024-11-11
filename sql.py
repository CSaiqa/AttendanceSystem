import mysql.connector

# Set up MySQL connection
conn = mysql.connector.connect(
    host="localhost",
    user="Cse299",
    password="12345",
    database="attendance_system"
)
cursor = conn.cursor()

# Function to add a new student
def add_student(student_id, name):
    try:
        query = "INSERT INTO Students (student_id, name) VALUES (%s, %s)"
        cursor.execute(query, (student_id, name))
        conn.commit()
        print(f"Student {name} with ID {student_id} added successfully.")
    except mysql.connector.Error as err:
        print(f"Error: {err}")

# Main program to prompt user input for adding students
if __name__ == "__main__":
    while True:
        student_id = input("Enter Student ID (or type 'exit' to quit): ")
        if student_id.lower() == 'exit':
            break
        name = input("Enter Student Name: ")
        add_student(student_id, name)

    # Close the database connection
    conn.close()




 




