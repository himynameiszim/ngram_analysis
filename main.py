import os
import nltk
from nltk import bigrams, trigrams
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import string
import shutil  # For removing the directory
import re

# Download nltk data
nltk.download('punkt')

# Preprocess input filenames
def preprocess_data(directory):
    for filename in os.listdir(directory):
        # Replace 'tm_family' with 'tm-family' and 'tm_friends' with 'tm-friends'
        new_filename = filename.replace('tm_family', 'tm-family').replace('tm_friends', 'tm-friends')
    
        # Find the participant number (e.g., P1, P93) and format it with leading zeros
        new_filename = re.sub(r'P(\d+)', lambda match: f'{int(match.group(1)):03d}', new_filename)
    
        # Get the full path of the original and new filename
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
    
        # Rename the file
        os.rename(old_file, new_file)
        print(f'Renamed: {filename} -> {new_filename}')

# Function to remove punctuation from text
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# Parse filenames and extract details
def parse_filename(filename):
    parts = filename.split('_')
    contestant_id = parts[0]
    gender = parts[1]
    container = parts[2]
    word_count = int(parts[3].replace('.txt', ''))
    return contestant_id, gender, container, word_count

# Read all text files and group them by contestants
def read_text_files(directory):
    contestant_files = defaultdict(list)
    for file in os.listdir(directory):
        if file.endswith('.txt'):
            filepath = os.path.join(directory, file)
            contestant_id, gender, container, word_count = parse_filename(file)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read().lower().replace('\n', ' ')
                text = remove_punctuation(text)  # Remove punctuation from the text
            contestant_files[contestant_id].append((gender, container, word_count, text))
    return contestant_files

# Tokenize text and extract n-grams
def extract_ngrams(text, n):
    tokens = [word for word in nltk.word_tokenize(text) if word.isalnum()]
    if n == 2:
        return list(bigrams(tokens))
    elif n == 3:
        return list(trigrams(tokens))
    return []

# Generate per-contestant CSV files
def generate_csv_files(contestant_id, contestant_data, n, output_dir):
    gender, containers = contestant_data[0][0], [data[1] for data in contestant_data]
    total_ngrams = Counter()
    container_ngrams = defaultdict(Counter)

    # Process each container and count n-grams
    for _, container, _, text in contestant_data:
        ngrams = extract_ngrams(text, n)
        total_ngrams.update(ngrams)
        container_ngrams[container].update(ngrams)
    
    # Calculate statistics
    total_ngram_count = sum(total_ngrams.values())
    rows = []
    for ngram, count in total_ngrams.items():
        normalized_freq = count / total_ngram_count
        shared_containers = sum(1 for container in container_ngrams if ngram in container_ngrams[container])
        row = {
            'ngram': ' '.join(ngram),
            'shared_containers': shared_containers,
            'total_count': count,
            'normalized_freq': normalized_freq
        }
        # Add container-specific data
        for container in containers:
            container_count = container_ngrams[container][ngram]
            container_freq = container_count / total_ngram_count if total_ngram_count > 0 else 0
            row[f'{container}_count'] = container_count
            row[f'{container}_freq'] = container_freq
        rows.append(row)

    # Create DataFrame and save
    columns = ['ngram', 'shared_containers', 'total_count', 'normalized_freq'] + \
              [f'{container}_count' for container in containers] + \
              [f'{container}_freq' for container in containers]
    df = pd.DataFrame(rows, columns=columns)
    output_file = os.path.join(output_dir, f'{contestant_id}_{"bigrams" if n == 2 else "trigrams"}.csv')
    df.to_csv(output_file, index=False)

# Main function
def main(input_dir, output_dir):
    # Remove the output directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Removed existing output directory: {output_dir}")
    
    # Recreate the output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    # Ask the user to choose between bigrams or trigrams
    ngram_choice = input("Choose n-gram type (2 for bigram, 3 for trigram): ")
    while ngram_choice not in ['2', '3']:
        ngram_choice = input("Invalid choice. Please enter 2 for bigram or 3 for trigram: ")
    ngram_choice = int(ngram_choice)

    contestant_files = read_text_files(input_dir)
    for contestant_id, contestant_data in contestant_files.items():
        # Generate the selected n-gram CSV file
        generate_csv_files(contestant_id, contestant_data, ngram_choice, output_dir)
        
        # print for debugging
        # print(f"Generated CSV file for contestant {contestant_id} ({'bigrams' if ngram_choice == 2 else 'trigrams'}).")

if __name__ == "__main__":
    input_dir = input("Enter the path to the input directory: ").strip()
    preprocess_data(input_dir)
    output_dir = input("Enter the path to the output directory: ").strip()
    main(input_dir, output_dir)
