import pandas as pd

# Define the label mapping
label_mapping = {
    'dokujo-tsushin': 0,
    'it-life-hack': 1,
    'kaden-channel': 2,
    'livedoor-homme': 3,
    'movie-enter': 4,
    'peachy': 5,
    'topic-news': 6,
    'smax': 7,
    'sports-watch': 8
}

# Read the CSV file
file_path = 'processed_data.csv'
df = pd.read_csv(file_path)

# Replace the labels with their corresponding numbers
df['label'] = df['label'].map(label_mapping)

# Save the modified DataFrame back to a CSV file
df.to_csv(file_path, index=False)