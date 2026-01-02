import pandas as pd
import os

def clean_data(input_folder, input_filename):
    # Construct the full path to the source file (dataset/dataset.csv)
    input_path = os.path.join(input_folder, input_filename)
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    # Load the dataset
    df = pd.read_csv(input_path)
    
    # Keep only text, category, and label
    df = df[['text', 'category', 'label']]
    
    # Change label: 'real' -> 1, 'fake' -> 0
    df['label'] = df['label'].map({'real': 1, 'fake': 0})
    
    # Get unique categories to split the data
    categories = df['category'].unique()
    
    for cat in categories:
        # Skip if category is NaN
        if pd.isna(cat):
            continue
            
        # Filter dataframe by the specific category
        cat_df = df[df['category'] == cat]
        
        # Format the filename: lowercase category name (e.g., Politics -> politics.csv)
        filename = f"{str(cat).lower().replace(' ', '_')}.csv"
        
        # Save directly back into the 'dataset' folder
        file_path = os.path.join(input_folder, filename)
        
        # Save to CSV without the index column
        cat_df.to_csv(file_path, index=False)
        print(f"Successfully exported {len(cat_df)} rows to {file_path}")

if __name__ == '__main__':
    # We tell the script the folder name and the file name separately
    clean_data('dataset', 'dataset.csv')