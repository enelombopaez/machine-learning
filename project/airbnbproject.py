import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Load the raw CSV file
def load_data(file_path):
    return pd.read_csv(file_path, delimiter=";", low_memory=False)

# Clean the dataset
def clean_data(df):
    columns_to_drop = [
        "Scrape ID", "Experiences Offered", "Thumbnail Url", "Medium Url", "XL Picture Url",
        "Host Thumbnail Url", "Host Picture Url", "Host URL", "Picture Url", "Calendar Updated",
        "Has Availability", "Calendar last Scraped", "Geolocation"
    ]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    
    price_columns = ["Price", "Weekly Price", "Monthly Price", "Security Deposit", "Cleaning Fee"]
    for col in price_columns:
        if col in df.columns:
            df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)
    
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    
    return df

# Exploratory Data Analysis (EDA)
def perform_eda(df):
    print("\n🔹 First 5 Rows of the Dataset:")
    print(df.head())

    print("\n🔹 Dataset Summary:")
    print(df.describe())

    print("\n🔹 Data Types:")
    print(df.dtypes)

    print("\n🔹 Missing Values Count:")
    print(df.isnull().sum())

    # Boxplot to visualize price distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df["Price"])
    plt.title("Price Distribution (Detecting Outliers)")
    plt.show()  # 🔹 Ensure the plot renders

    # Heatmap for numerical feature correlation
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] > 1:  # Ensure there are at least 2 numerical columns
        plt.figure(figsize=(12, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation (Numerical Only)")
        plt.show()  # 🔹 Ensure the heatmap renders
    else:
        print("\n⚠ No numerical columns available for correlation heatmap.")
    
# Feature Selection using Random Forest
def feature_selection(df, target_column="Price"):
    # Drop URL-like columns (ensure no URLs remain)
    url_columns = [col for col in df.columns if "url" in col.lower()]
    df = df.drop(columns=url_columns, errors="ignore")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Convert categorical variables to numerical values
    label_encoders = {}
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Train the RandomForest model for feature selection
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    selector = SelectFromModel(model, prefit=True)
    selected_features = X.columns[(selector.get_support())]

    print("\nSelected Features:", list(selected_features))
    return df[selected_features].join(y)

# Data Scaling
def scale_data(df):
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=["number"]).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
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

def feature_selection_and_engineering(df):
    # Ensure df is a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Keep relevant columns
    columns_to_keep = ["Last Scraped", "Street", "State", "Latitude", "Longitude", 
                       "Room Type", "Accommodates", "Bathrooms", "Bedrooms", 
                       "Monthly Price", "Cleaning Fee", "Host Since", "Price"]
    
    df = df[columns_to_keep].copy()

    # Convert "Host Since" to datetime, handling "Unknown" values
    df["Host Since"] = pd.to_datetime(df["Host Since"], errors="coerce")

    # Create new feature: Host Age (years since listing)
    df["Host Age"] = pd.to_datetime("today").year - df["Host Since"].dt.year

    # Fill missing values in Host Age with the median
    df["Host Age"].fillna(df["Host Age"].median(), inplace=True)

    # Drop the original "Host Since" column (no longer needed)
    df.drop(columns=["Host Since"], inplace=True)

    print(f"✅ Selected Features: {list(df.columns)}")
    return df

# Main function to execute the pipeline
def main():
    file_path = "/home/nelson/machine-learning/project/airbnb-listings-extract.csv"
    df = load_data(file_path)
    df_cleaned = clean_data(df)
    perform_eda(df_cleaned)

    # Feature selection & engineering
    df_selected = feature_selection_and_engineering(df_cleaned)

    # Ensure "Price" is included before scaling
    df_scaled = scale_data(df_selected)

    # Save cleaned and processed data
    cleaned_file_path = "airbnb_cleaned.csv"
    df_scaled.to_csv(cleaned_file_path, index=False)
    print(f"✅ Cleaned and processed data saved to {cleaned_file_path}")

    # Split data (make sure "Price" exists)
    X_train, X_test, y_train, y_test = split_data(df_scaled, target_column="Price")
    
    # Save train/test sets
    X_train.to_csv("X_train.csv", index=False)
    X_test.to_csv("X_test.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)
    print("✅ Train and test datasets saved.")

if __name__ == "__main__":
    main()