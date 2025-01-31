import pandas as pd
import numpy as np

# Generate the random Netflix dataset
def generate_netflix_data():
    data = {
        'CustomerID': np.arange(2001, 2201),
        'SubscriptionType': np.random.choice(['Basic', 'Standard', 'Premium'], 200),
        'MonthlyFee': np.random.choice([8.99, 13.99, 17.99], 200),
        'TotalWatchHours': np.random.randint(50, 700, 200),
        'NumDevices': np.random.randint(1, 6, 200),
        'HasAutoRenew': np.random.choice([0, 1], 200),
        'LastPaymentMissed': np.random.choice([0, 1], 200),
        'AvgWatchTimePerDay': np.random.uniform(0.5, 6.5, 200),
        'CancelledBefore': np.random.choice([0, 1], 200)  # Churn column (0: no, 1: yes)
    }
    df = pd.DataFrame(data)
    df.rename(columns={'CancelledBefore': 'Churn'}, inplace=True)  # Rename column to 'Churn'
    
    # Save the DataFrame as a CSV file
    df.to_csv('netflix_data.csv', index=False)
    print("Dataset saved as 'netflix_data2.csv'")

# Call the function to generate and save the dataset
generate_netflix_data()
