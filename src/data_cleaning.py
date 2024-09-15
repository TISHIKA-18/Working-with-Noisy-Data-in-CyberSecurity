import pandas as pd

# Load the dataset
df = pd.read_csv('data/UNSW-NB15.csv')

# 1. Handling Missing Data
df.fillna(method='ffill', inplace=True)

# 2. Removing Duplicates
df.drop_duplicates(inplace=True)

# 3. Standardizing Timestamps
df['Start_time'] = pd.to_datetime(df['Start_time'], utc=True)
df['Last_time'] = pd.to_datetime(df['Last_time'], utc=True)

# Save the cleaned data
df.to_csv('data/cleaned_UNSW-NB15.csv', index=False)

print("Data cleaning completed!")
