import os
import nltk
from nltk import bigrams, trigrams
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import string
import shutil  # For removing the directory
import re
import matplotlib.pyplot as plt

# Download nltk data
# nltk.download('punkt')
nltk.download('punkt_tab')

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

# Parse filenames
def parse_filename(filename):
    parts = filename.split('_')
    # contestant_id = parts[0]
    # gender = parts[1]
    # container = parts[2]
    # word_count = int(parts[3].replace('.txt', ''))
    return parts[0], parts[1], parts[2], int(parts[3].replace('.txt', ''))

# Read and group text files
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

# Extract n-grams
def extract_ngrams(text, n):
    tokens = [word for word in nltk.word_tokenize(text) if word.isalnum()]
    return list(bigrams(tokens)) if n == 2 else list(trigrams(tokens))

# Generate CSV files for n-grams
def generate_csv_files(contestant_id, contestant_data, n, output_dir):
    containers = [data[1] for data in contestant_data]
    total_ngrams = Counter()
    container_ngrams = defaultdict(Counter)

    # Calculate n-grams and counts
    for _, container, _, text in contestant_data:
        ngrams = extract_ngrams(text, n)
        total_ngrams.update(ngrams)
        container_ngrams[container].update(ngrams)
    
    # Calculate statistics
    total_ngram_count = sum(total_ngrams.values())
    container_counts = {container: sum(container_ngrams[container].values()) for container in containers}
    rows = []
    for ngram, count in total_ngrams.items():
        normalized_freq = count / total_ngram_count
        shared_containers = sum(1 for container in container_ngrams if ngram in container_ngrams[container])
        row = {'ngram': ' '.join(ngram), 'total_count': count, 'normalized_freq': normalized_freq}

        # Add container-specific data
        for container in containers:
            container_count = container_ngrams[container][ngram]
            container_freq = container_count / container_counts[container] if container_counts[container] > 0 else 0
            row[f'{container}_count'] = container_count
            row[f'{container}_freq'] = container_freq
        rows.append(row)

    # Prepare columns
    columns = ['ngram', 'total_count', 'normalized_freq'] + \
              [f'{container}_count' for container in containers] + \
              [f'{container}_freq' for container in containers]
    output_file = os.path.join(output_dir, f'{contestant_id}_{"bigrams" if n == 2 else "trigrams"}.csv')
    pd.DataFrame(rows, columns=columns).to_csv(output_file, index=False)
    print(f"N-gram CSV file saved: {output_file}")

# Generate Z-Score CSV with Sigma Calculation
def generate_z_score_csv_with_sigma(contestant_id, contestant_data, n, output_dir):
    containers = [data[1] for data in contestant_data]
    total_ngrams = Counter()
    container_ngrams = defaultdict(Counter)
    for _, container, _, text in contestant_data:
        ngrams = extract_ngrams(text, n)
        total_ngrams.update(ngrams)
        container_ngrams[container].update(ngrams)
    total_ngram_count = sum(total_ngrams.values())
    container_counts = {container: sum(container_ngrams[container].values()) for container in containers}
    rows = []
    for ngram, count in total_ngrams.items():
        container_specific_counts = [container_ngrams[container][ngram] for container in containers]
        global_sigma = np.std(list(total_ngrams.values())) if len(total_ngrams) > 1 else 0
        global_z_score = (count - np.mean(list(total_ngrams.values()))) / global_sigma if global_sigma > 0 else 0
        container_z_scores = [
            (container_ngrams[container][ngram] - np.mean(container_specific_counts)) / np.std(container_specific_counts)
            if len(container_specific_counts) > 1 else 0
            for container in containers
        ]
        rows.append([ngram, global_z_score, global_sigma] + container_z_scores)
    columns = ['ngram', 'global_z_score', 'global_sigma'] + [f'z_score_{container}' for container in containers]
    z_score_file = os.path.join(output_dir, f"{contestant_id}_{'bigrams' if n == 2 else 'trigrams'}_z_score.csv")
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(z_score_file, index=False)
    print(f"Z-score file saved: {z_score_file}")
    # Return sigma for visualization
    return global_sigma
# Plot Sigma for All Contestants (Sorted X-Axis)
def plot_sigma(output_dir, sigma_data, ngram_type):
    # Sort sigma_data by contestant keys (ascending order)
    sorted_data = dict(sorted(sigma_data.items()))
    contestants = list(sorted_data.keys())
    sigmas = list(sorted_data.values())
    plt.figure(figsize=(14, 6))  # Larger figure size for readability
    plt.plot(contestants, sigmas, marker='o', linestyle='-', color='skyblue', label="Sigma (Standard Deviation)")
    # Annotate points if fewer than 20 contestants, otherwise skip annotations
    if len(contestants) <= 20:
        for i, sigma in enumerate(sigmas):
            plt.text(contestants[i], sigma, f'{sigma:.2f}', fontsize=9, ha='center', va='bottom')
    # Adjust x-axis for large datasets
    if len(contestants) > 50:
        # Skip some labels for better readability
        skip = max(1, len(contestants) // 20)  # Show approximately 20 labels
        plt.xticks(ticks=range(0, len(contestants), skip), 
                   labels=[contestants[i] for i in range(0, len(contestants), skip)], rotation=45)
    else:
        plt.xticks(rotation=90)
    plt.xlabel("Contestants")
    plt.ylabel("Sigma (Standard Deviation)")
    plt.title(f"Sigma (Standard Deviation) for {ngram_type.capitalize()} Across Contestants")
    plt.legend()
    plt.tight_layout()
    # Save the plot to the output directory
    plot_file = os.path.join(output_dir, f"{ngram_type}_sigma_plot_sorted.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Sorted Sigma plot saved: {plot_file}")

# Main function
def main(input_dir, output_dir):
    # Remove the output directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
       
    os.makedirs(output_dir, exist_ok=True)

    # Ask the user to choose between bigrams or trigrams
    ngram_choice = input("Choose n-gram type (2 for bigram, 3 for trigram): ").strip()
    while ngram_choice not in ['2', '3']:
        ngram_choice = input("Invalid choice. Please enter 2 for bigram or 3 for trigram: ").strip()
    ngram_choice = int(ngram_choice)
    ngram_type = 'bigrams' if ngram_choice == 2 else 'trigrams'

    contestant_files = read_text_files(input_dir)
    sigma_data = {}

    for contestant_id, contestant_data in contestant_files.items():
        # Generate the selected n-gram CSV file
        generate_csv_files(contestant_id, contestant_data, ngram_choice, output_dir)
        
        sigma = generate_z_score_csv_with_sigma(contestant_id, contestant_data, ngram_choice, output_dir)
        sigma_data[contestant_id] = sigma

    # Plot sigma data
    plot_sigma(output_dir, sigma_data, ngram_type)
        # print for debugging
        # print(f"Generated CSV file for contestant {contestant_id} ({'bigrams' if ngram_choice == 2 else 'trigrams'}).")

if __name__ == "__main__":
    input_dir = input("Enter the path to the input directory: ").strip()
    preprocess_data(input_dir)
    output_dir = input("Enter the path to the output directory: ").strip()
    main(input_dir, output_dir)
