import pandas as pd
from sklearn.model_selection import train_test_split

# Load the raw CSV file
def load_data(file_path):
    return pd.read_csv(file_path, delimiter=";", low_memory=False)

# Clean the dataset
def clean_data(df):
    # Drop unnecessary columns
    columns_to_drop = [
        "Scrape ID", "Experiences Offered", "Thumbnail Url", "Medium Url", "XL Picture Url",
        "Host Thumbnail Url", "Host Picture Url", "Host URL", "Picture Url", "Calendar Updated",
        "Has Availability", "Calendar last Scraped", "Geolocation"
    ]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

    # Convert price-related columns to numeric
    price_columns = ["Price", "Weekly Price", "Monthly Price", "Security Deposit", "Cleaning Fee"]
    for col in price_columns:
        if col in df.columns:
            df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)
    
    # Fill missing values
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    
    return df

# Split data into training and testing sets
def split_data(df, target_column="Price", test_size=0.2):
    if target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df
        y = None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# Main function to execute the pipeline
def main():
    file_path = "/home/nelson/machine-learning/project/airbnb-listings-extract.csv"  # Update this if needed
    df = load_data(file_path)
    df_cleaned = clean_data(df)
    
    # Save cleaned data
    cleaned_file_path = "/home/nelson/machine-learning/project/airbnb_cleaned.csv"
    df_cleaned.to_csv(cleaned_file_path, index=False)
    print(f"Cleaned data saved to {cleaned_file_path}")
    
    # Split and save train/test sets
    X_train, X_test, y_train, y_test = split_data(df_cleaned)
    X_train.to_csv("/home/nelson/machine-learning/project/X_train.csv", index=False)
    X_test.to_csv("/home/nelson/machine-learning/project/X_test.csv", index=False)
    if y_train is not None:
        y_train.to_csv("/home/nelson/machine-learning/project/y_train.csv", index=False)
        y_test.to_csv("/home/nelson/machine-learning/project/y_test.csv", index=False)
    print("Train and test datasets saved.")

if __name__ == "__main__":
    main()
