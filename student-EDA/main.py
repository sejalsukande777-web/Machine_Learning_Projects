import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Function Name : load_data()
# Description   : Loads CSV dataset into Pandas DataFrame
# Input         : Filename
# Output        : DataFrame
# ---------------------------------------------------------------

def load_data(filename):
    df = pd.read_csv(filename)

    print("\n------------------------------------------------")
    print("Dataset Loaded Successfully")
    print("------------------------------------------------\n")

    return df

# ---------------------------------------------------------------
# Function Name : display_basic_info()
# Description   : Displays complete dataset information
#                 including shape, columns and null values
# Input         : DataFrame
# Output        : None
# ---------------------------------------------------------------

def display_basic_info(df):

    print("\n================================================")
    print("Complete Dataset")
    print("================================================")
    print(df)

    print("\n================================================")
    print("First 5 Records")
    print("================================================")
    print(df.head())

    print("\n================================================")
    print("Dataset Shape")
    print("================================================")
    print(df.shape)

    print("\n================================================")
    print("Column Names")
    print("================================================")
    print(df.columns)

    print("\n================================================")
    print("Dataset Information")
    print("================================================")
    print(df.info())

    print("\n================================================")
    print("Missing Values")
    print("================================================")
    print(df.isnull().sum())


# ---------------------------------------------------------------
# Function Name : sql_like_analysis()
# Description   : Performs SQL-like analytics operations
#                 using Pandas
# Input         : DataFrame
# Output        : None
# ---------------------------------------------------------------

def sql_like_analysis(df):

    print("\n================================================")
    print("Course Wise Student Count")
    print("================================================")
    print(df.groupby("Course")["StudentID"].count())

    print("\n================================================")
    print("Average Marks by Course")
    print("================================================")
    print(df.groupby("Course")["Marks"].mean())

    print("\n================================================")
    print("City Wise Student Count")
    print("================================================")
    print(df.groupby("City")["StudentID"].count())

    print("\n================================================")
    print("Gender Distribution")
    print("================================================")
    print(df.groupby("Gender")["StudentID"].count())

    print("\n================================================")
    print("Average Attendance by Batch")
    print("================================================")
    print(df.groupby("Batch")["Attendance"].mean())

    print("\n================================================")
    print("Fees Collection by Course")
    print("================================================")
    print(df.groupby("Course")["FeesPaid"].sum())

    print("\n================================================")
    print("Top Performing Students")
    print("================================================")
    print(df[df["Marks"] >= 80][["Name", "Course", "Marks"]])

    print("\n================================================")
    print("Low Attendance Students")
    print("================================================")
    print(df[df["Attendance"] < 75][["Name", "Batch", "Attendance"]])

    print("\n================================================")
    print("Highest Marks")
    print("================================================")
    print(df["Marks"].max())

    print("\n================================================")
    print("Lowest Marks")
    print("================================================")
    print(df["Marks"].min())

    print("\n================================================")
    print("Complete Course Wise Analytics Report")
    print("================================================")

    report = df.groupby("Course").agg(
        Total_Students=("StudentID", "count"),
        Average_Marks=("Marks", "mean"),
        Average_Attendance=("Attendance", "mean"),
        Total_Fees=("FeesPaid", "sum"),
        Highest_Marks=("Marks", "max"),
        Lowest_Marks=("Marks", "min")
    )

    print(report)

# ---------------------------------------------------------------
# Function Name : create_visualizations()
# Description   : Creates multiple visualizations for
#                 analytics and dashboard representation
# Input         : DataFrame
# Output        : Graphical Visualization
# ---------------------------------------------------------------

def create_visualizations(df):

    # -----------------------------------------------------------
    # Course Wise Student Count
    # -----------------------------------------------------------

    course_count = df.groupby("Course")["StudentID"].count()

    plt.figure(figsize=(8, 5))

    course_count.plot(kind="bar")

    plt.title("Course Wise Student Count")
    plt.xlabel("Course")
    plt.ylabel("Number of Students")

    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------
    # Average Marks by Course
    # -----------------------------------------------------------

    avg_marks = df.groupby("Course")["Marks"].mean()

    plt.figure(figsize=(8, 5))

    avg_marks.plot(kind="bar")

    plt.title("Average Marks by Course")
    plt.xlabel("Course")
    plt.ylabel("Average Marks")

    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------
    # Gender Distribution
    # -----------------------------------------------------------

    gender_count = df.groupby("Gender")["StudentID"].count()

    plt.figure(figsize=(7, 7))

    gender_count.plot(kind="pie", autopct="%1.1f%%")

    plt.title("Gender Distribution")
    plt.ylabel("")

    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------
    # City Wise Student Count
    # -----------------------------------------------------------

    city_count = df.groupby("City")["StudentID"].count()

    plt.figure(figsize=(8, 5))

    city_count.plot(kind="bar")

    plt.title("City Wise Student Count")
    plt.xlabel("City")
    plt.ylabel("Number of Students")

    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------
    # Attendance Analysis
    # -----------------------------------------------------------

    attendance = df.sort_values("Attendance", ascending=False)

    plt.figure(figsize=(9, 5))

    plt.bar(attendance["Name"], attendance["Attendance"])

    plt.title("Student Attendance Analysis")
    plt.xlabel("Student Name")
    plt.ylabel("Attendance")

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------
    # Fees Collection by Course
    # -----------------------------------------------------------

    fees = df.groupby("Course")["FeesPaid"].sum()

    plt.figure(figsize=(8, 5))

    fees.plot(kind="bar")

    plt.title("Fees Collection by Course")
    plt.xlabel("Course")
    plt.ylabel("Total Fees")

    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------
    # Marks Distribution
    # -----------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.hist(df["Marks"], bins=5)

    plt.title("Marks Distribution")
    plt.xlabel("Marks")
    plt.ylabel("Number of Students")

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------
# Function Name : main()
# Description   : Entry point function of application
# Input         : None
# Output        : None
# ---------------------------------------------------------------
def main():

    filename = "student.csv"

    df = load_data(filename)

    display_basic_info(df)

    sql_like_analysis(df)

    create_visualizations(df)


# ---------------------------------------------------------------
# Application Starter
# ---------------------------------------------------------------
if __name__ == "__main__":
    main()