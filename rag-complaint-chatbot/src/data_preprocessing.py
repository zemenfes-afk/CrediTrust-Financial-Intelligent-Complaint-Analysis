import pandas as pd
import re
import os

# Configuration
RAW_DATA_PATH = "../data/raw/complaints.csv"  # Ensure you download the CFPB CSV here
PROCESSED_DATA_PATH = "../data/processed/filtered_complaints.csv"

TARGET_PRODUCTS = [
    "Credit card or prepaid card",
    "Mortgage",  # Note: Verify specific dataset labels. The prompt asks for:
    # Credit Card, Personal Loan, Savings, Money Transfer.
    # Adjusting strictly to prompt:
]

# Mapping strictly to the prompt's 5 categories based on standard CFPB taxonomy
# Note: CFPB taxonomy changes over time. These are standard mappings.
PRODUCT_MAPPING = {
    "Credit card": "Credit Card",
    "Credit card or prepaid card": "Credit Card",
    "Payday loan, title loan, or personal loan": "Personal Loan",
    "Checking or savings account": "Savings Account",
    "Money transfer, virtual currency, or money service": "Money Transfers",
    "Money transfers": "Money Transfers"
}


def clean_text(text):
    """
    Cleans the complaint narrative text.
    1. Lowercase
    2. Remove "XXXX" (anonymized data in CFPB)
    3. Remove special characters
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    # Remove CFPB redaction markers
    text = re.sub(r'x{2,}', '', text)
    # Remove boilerplate "I am writing to..." if consistently present,
    # but for now we stick to general cleaning to avoid over-pruning.
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def preprocess_data():
    print("Loading data...")
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Please place the CFPB dataset at {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)

    # 1. Filter for rows with narratives
    df = df.dropna(subset=['Consumer complaint narrative'])

    # 2. Filter by Product Category
    # We create a new uniform column 'category' based on the mapping
    df['category'] = df['Product'].map(PRODUCT_MAPPING)
    df = df.dropna(subset=['category'])

    # 3. Clean Text
    print("Cleaning narratives...")
    df['cleaned_narrative'] = df['Consumer complaint narrative'].apply(clean_text)

    # 4. Filter empty cleaned narratives
    df = df[df['cleaned_narrative'].str.len() > 10]

    # Save
    print(f"Saving {len(df)} records to {PROCESSED_DATA_PATH}...")
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print("Preprocessing complete.")


if __name__ == "__main__":
    preprocess_data()