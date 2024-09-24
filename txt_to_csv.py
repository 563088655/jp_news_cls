import os
import pandas as pd

# detaset path
root_dir = 'text'

# Initialize
title_list = []
content_list = []
label_list = []

# Traverse all category folders
for category in os.listdir(root_dir):
    category_path = os.path.join(root_dir, category)
    
    # Ensure it is a directory
    if os.path.isdir(category_path):
        # Traverse all files in the category folder
        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)
            
            # Read the content of each file
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                
                # Check if the file has enough lines
                if len(lines) >= 4:
                    title = lines[2].strip()  # Get the title
                    
                    # Get the content, starting from the fourth line, merge all remaining lines
                    content = ''.join(lines[3:]).strip()
                    
                    # Store in lists
                    title_list.append(title)
                    content_list.append(content)
                    label_list.append(category)

# Create DataFrame
df = pd.DataFrame({
    'label': label_list,
    'title': title_list,
    'content': content_list
})

# Display the first few rows of data
print(df.head())

# Save as CSV file
df.to_csv('processed_data.csv', index=False, encoding='utf-8')

# the dataset is from here: https://www.rondhuit.com/download.html
# 記事ファイルは以下のフォーマットにしたがって作成されています：
# the format of the article file is as follows:

# １行目：記事のURL
# the first line: URL of the article
# ２行目：記事の日付
# the second line: date of the article
# ３行目：記事のタイトル
# the third line: title of the article
# ４行目以降：記事の本文
# from the fourth line: the body of the article