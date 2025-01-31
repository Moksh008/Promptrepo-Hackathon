import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate a random dataset with 1000 samples
num_samples = 1000

# Define column names and their data types for a food delivery service
columns = ['CustomerID', 'SubscriptionType', 'OrderFrequency', 'TotalSpend', 
           'NumOrders', 'DeliveryPreference', 'AvgOrderValue', 'CustomerRating']

# Create the dataset
df = pd.DataFrame({
    'CustomerID': range(2001, 2001 + num_samples),
    'SubscriptionType': np.random.choice(['Basic', 'Premium'], num_samples),  # Customer's subscription type
    'OrderFrequency': np.random.choice(['Daily', 'Weekly', 'Monthly'], num_samples),  # How often they order
    'TotalSpend': np.random.randint(50, 5000, size=num_samples),  # Total money spent in the service
    'NumOrders': np.random.randint(1, 200, size=num_samples),  # Total number of orders made
    'DeliveryPreference': np.random.choice(['Home Delivery', 'Pick-up'], num_samples),  # Delivery preference
    'AvgOrderValue': np.round(np.random.uniform(10, 150, size=num_samples), 2),  # Average order value in USD
    'CustomerRating': np.round(np.random.uniform(1, 5, size=num_samples), 2),  # Rating given by the customer (1-5)
})

# Display the first few rows of the dataset
print(df.head())

# Save the dataset to a CSV file
df.to_csv('food_delivery_service_data.csv', index=False)

# Inform the user
print("Dataset saved as 'food_delivery_service_data.csv'.")
