import os
import xml.etree.ElementTree as ET
import nltk
from nltk import bigrams, trigrams
from collections import Counter
import pandas as pd
import numpy as np
import string

# Download NLTK data if not already available
nltk.download('punkt')

# Function to remove metadata (teiHeader)
def remove_metadata_from_xml(tree):
    root = tree.getroot()
    for tei_header in root.findall('teiHeader'):
        root.remove(tei_header)
    return tree

# Function to extract text recursively and write to a file
def extract_text_to_file(element, file):
    for child in element:
        if child.text and child.text.strip():
            file.write(f"{child.tag}: {child.text.strip()}\n")
        extract_text_to_file(child, file)

# Function to process a file
def process_file(input_file, output_file):
    with open(input_file, "r") as infile:
        lines = infile.readlines()

    # Extract text after ':' and concatenate
    result = ""
    for line in lines:
        if ':' in line:
            _, text = line.split(':', 1)
            result += text.strip() + " "

    # Write concatenated text to output file
    with open(output_file, "w") as outfile:
        outfile.write(result.strip())

# Function to process the entire folder structure
def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(input_folder):
        # Recreate directory structure in output folder
        relative_path = os.path.relpath(root, input_folder)
        current_output_folder = os.path.join(output_folder, relative_path)
        if not os.path.exists(current_output_folder):
            os.makedirs(current_output_folder)

        for file_name in files:
            if file_name.endswith(".xml"):
                input_file_path = os.path.join(root, file_name)
                temp_output_file = os.path.join(current_output_folder, f"{file_name}.tmp")

                # Remove .xml from the output file name and add .txt
                output_file_name = file_name.replace(".xml", "") + ".txt"
                final_output_file = os.path.join(current_output_folder, output_file_name)

                # Remove metadata and extract text
                tree = ET.parse(input_file_path)
                tree = remove_metadata_from_xml(tree)

                # Write the cleaned XML to a temp file
                temp_xml_path = os.path.join(current_output_folder, f"temp_{file_name}")
                tree.write(temp_xml_path)

                with open(temp_output_file, "w") as temp_file:
                    root_element = tree.getroot()
                    extract_text_to_file(root_element, temp_file)

                process_file(temp_output_file, final_output_file)

                # Clean up temporary files
                os.remove(temp_output_file)
                os.remove(temp_xml_path)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# Extract n-grams (bigrams in this case)
def extract_bigrams(text):
    tokens = [word for word in nltk.word_tokenize(text) if word.isalnum()]
    return list(bigrams(tokens))

# Extract n-grams (trigrams in this case)
def extract_trigrams(text):
    tokens = [word for word in nltk.word_tokenize(text) if word.isalnum()]
    return list(trigrams(tokens))

# Function to calculate z-scores for bigrams
def calculate_z_scores(total_ngrams):
    total_ngram_count = sum(total_ngrams.values())
    global_mean = np.mean(list(total_ngrams.values()))
    global_sigma = np.std(list(total_ngrams.values())) if len(total_ngrams) > 1 else 0

    rows = []
    for ngram, count in total_ngrams.items():
        z_score = (count - global_mean) / global_sigma if global_sigma > 0 else 0
        rows.append({
            "ngram": ' '.join(ngram),
            "count": count,
            "z_score": z_score
        })

    return rows

# Function to process all files in the input folder
def process_corpus_folder_bigram(input_folder, output_file):
    total_ngrams = Counter()

    # Read and process each file
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(input_folder, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().lower().replace('\n', ' ')
                text = remove_punctuation(text)

            # Extract bigrams and update the total ngram counter
            bigrams_list = extract_bigrams(text)
            total_ngrams.update(bigrams_list)

    # Calculate z-scores
    rows = calculate_z_scores(total_ngrams)

    # Save results to a CSV file
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Bigrams and their z-scores saved to: {output_file}")

# Function to process all files in the input folder
def process_corpus_folder_trigram(input_folder, output_file):
    total_ngrams = Counter()

    # Read and process each file
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(input_folder, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().lower().replace('\n', ' ')
                text = remove_punctuation(text)

            # Extract trigrams and update the total ngram counter
            trigrams_list = extract_trigrams(text)
            total_ngrams.update(trigrams_list)

    # Calculate z-scores
    rows = calculate_z_scores(total_ngrams)

    # Save results to a CSV file
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Trigrams and their z-scores saved to: {output_file}")

# Main execution
if __name__ == "__main__":
    input_folder = input("Enter the path to the corpus input directory: ").strip()
    output_file = input("Enter the path to the output file: ").strip()

    process_folder(input_folder, input_folder)
    print(f"Preprocessed files are saved in '{input_folder}'.")

    process_corpus_folder_bigram(input_folder, output_file)
    process_corpus_folder_trigram(input_folder, output_file)