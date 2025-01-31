import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate a random dataset with 1000 samples
num_samples = 1000

# Define column names and their data types for a music streaming service
columns = ['CustomerID', 'SubscriptionType', 'MonthlyFee', 'TotalListenHours', 
           'NumDevices', 'HasAutoRenew', 'LastPaymentMissed', 
           'AvgListenTimePerDay']

# Create the dataset
df = pd.DataFrame({
    'CustomerID': range(2001, 2001 + num_samples),
    'SubscriptionType': np.random.choice(['Free', 'Premium'], num_samples),
    'MonthlyFee': np.random.choice([0, 9.99], num_samples),  # 0 for Free, 9.99 for Premium
    'TotalListenHours': np.random.randint(10, 1000, size=num_samples),
    'NumDevices': np.random.randint(1, 6, size=num_samples),
    'HasAutoRenew': np.random.choice([0, 1], num_samples),
    'LastPaymentMissed': np.random.choice([0, 1], num_samples),
    'AvgListenTimePerDay': np.round(np.random.uniform(0.5, 10.0, size=num_samples), 2),
})

# Display the first few rows of the dataset
print(df.head())

# Save the dataset to a CSV file
df.to_csv('spotify_no_churn.csv', index=False)

# Inform the user
print("Dataset saved as 'spotify_no_churn.csv'.")
