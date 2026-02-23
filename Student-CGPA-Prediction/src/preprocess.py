import pandas as pd

def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Drop ID
    df.drop(columns=['Student_ID'], inplace=True)

    # RENAME CGPA → Current_CGPA (THIS IS CRITICAL)
    df.rename(columns={'CGPA': 'Current_CGPA'}, inplace=True)

    subject_cols = [
        'Subject1_Marks', 'Subject2_Marks', 'Subject3_Marks',
        'Subject4_Marks', 'Subject5_Marks'
    ]

    df['Avg_Subject_Marks'] = df[subject_cols].mean(axis=1)
    df.drop(columns=subject_cols, inplace=True)

    return df