import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset with engineered features
df = pd.read_csv('data/engineered_UNSW-NB15.csv')

# Selecting relevant numeric columns
features = ['Source_Port', 'Destination_Port', 'Duration']

# Standardizing the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

# Applying PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)

# Adding PCA components to the dataset
df['PCA1'] = principal_components[:, 0]
df['PCA2'] = principal_components[:, 1]

# Save the reduced data
df.to_csv('data/reduced_UNSW-NB15.csv', index=False)

print("Dimensionality reduction completed!")
