import pandas as pd

# Load dataset
file_path = "Walmart.csv"  # Adjust the path if needed
df = pd.read_csv(file_path)

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Remove exact duplicate rows (if any)
df.drop_duplicates(inplace=True)

# Fill missing values using forward fill method
df.fillna(method='ffill', inplace=True)

# Save cleaned data
df.to_csv("cleaned_walmart.csv", index=False)

# Print final dataset info
print("\n✅ Data Cleaning Complete! Cleaned dataset saved as 'cleaned_walmart.csv'.")
print(df.info())