import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter

# Load the dataset
df = pd.read_csv('data/reduced_UNSW-NB15.csv')

# Separating features and target variable
X = df.drop('Label', axis=1)
y = df['Label']

# Applying SMOTE
smote = SMOTE(sampling_strategy='auto')
X_res, y_res = smote.fit_resample(X, y)

# Save the balanced data
resampled_data = pd.concat([pd.DataFrame(X_res), pd.DataFrame(y_res, columns=['Label'])], axis=1)
resampled_data.to_csv('data/balanced_UNSW-NB15.csv', index=False)

print(f"Data after SMOTE: {Counter(y_res)}")
