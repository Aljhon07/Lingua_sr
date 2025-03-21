import os
import pandas as pd
from sklearn.model_selection import train_test_split

path = os.path.abspath(os.curdir) + "/commonvoice/en/"

def split_data(tsv_file, output_dir, train_size=0.8, dev_size=0.1):
    print(f"Loading data from {tsv_file}...")
    # Load metadata
    data = pd.read_csv(tsv_file, sep='\t')
    print("Data loaded successfully.")
    
    # Split data into train, dev, and test sets
    print("Splitting data into train, dev, and test sets...")
    train_data, temp_data = train_test_split(data, train_size=train_size, random_state=42)
    dev_data, test_data = train_test_split(temp_data, train_size=dev_size/(1-train_size), random_state=42)
    print("Data split successfully.")
    
    # Save splits
    print(f"Saving train data to {os.path.join(output_dir, 'train.tsv')}...")
    train_data.to_csv(os.path.join(output_dir, 'train.tsv'), sep='\t', index=False)
    print(f"Saving dev data to {os.path.join(output_dir, 'dev.tsv')}...")
    dev_data.to_csv(os.path.join(output_dir, 'dev.tsv'), sep='\t', index=False)
    print(f"Saving test data to {os.path.join(output_dir, 'test.tsv')}...")
    test_data.to_csv(os.path.join(output_dir, 'test.tsv'), sep='\t', index=False)
    print("Data saved successfully.")

if __name__ == "__main__":
    tsv_file = path + 'validated.tsv'
    output_dir = path
    split_data(tsv_file, output_dir)
