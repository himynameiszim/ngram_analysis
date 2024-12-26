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

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def extract_bigrams(text):
    tokens = [word for word in nltk.word_tokenize(text) if word.isalnum()]
    return list(bigrams(tokens))

def extract_trigrams(text):
    tokens = [word for word in nltk.word_tokenize(text) if word.isalnum()]
    return list(trigrams(tokens))

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

def detect_file_type(file_path):
    """Detect whether an XML file is tagged or untagged."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Check for specific tags to classify file type
        if root.find('teiHeader') is not None:
            return "tagged"
        elif root.find('header') is not None:
            return "untagged"
        else:
            return "unknown"
    except Exception as e:
        print(f"Error detecting file type for {file_path}: {e}")
        return "unknown"

def process_tagged_file(file_path):
    """Process a tagged XML file."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Remove <teiHeader>
        for tei_header in root.findall('teiHeader'):
            root.remove(tei_header)

        text_content = []
        for elem in root.iter():
            if elem.text:
                text_content.append(elem.text.strip())

        return ' '.join(text_content)
    except Exception as e:
        print(f"Error processing tagged file {file_path}: {e}")
        return ""

def process_untagged_file(file_path):
    """Process an untagged XML file."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Remove <header> if present
        header = root.find('header')
        if header is not None:
            root.remove(header)

        # Extract text from <body>
        body = root.find('body')
        if body is None:
            return ""

        text_content = []
        for elem in body.iter():
            if elem.text:
                text_content.append(elem.text.strip())

        return ' '.join(text_content)
    except Exception as e:
        print(f"Error processing untagged file {file_path}: {e}")
        return ""

def process_and_save_text(input_folder, output_folder, ngram_output_file, ngram_type="bigram"):
    """Process all files, extract text, and calculate n-grams."""
    os.makedirs(output_folder, exist_ok=True)
    total_ngrams = Counter()

    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)

        if os.path.isfile(file_path) and file_name.endswith('.xml'):
            file_type = detect_file_type(file_path)

            # Process based on file type
            if file_type == "tagged":
                text = process_tagged_file(file_path)
            elif file_type == "untagged":
                text = process_untagged_file(file_path)
            else:
                print(f"Skipping unknown file type: {file_path}")
                continue

            # Save processed text to a .txt file
            output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)

            # Clean text and extract n-grams
            text = remove_punctuation(text.lower().replace('\n', ' '))
            if ngram_type == "bigram":
                ngrams_list = extract_bigrams(text)
            elif ngram_type == "trigram":
                ngrams_list = extract_trigrams(text)

            total_ngrams.update(ngrams_list)

    # Calculate Z-scores and save to CSV
    rows = calculate_z_scores(total_ngrams)
    df = pd.DataFrame(rows)
    df.to_csv(ngram_output_file, index=False)
    print(f"{ngram_type.capitalize()}s and their z-scores saved to: {ngram_output_file}")

if __name__ == "__main__":
    input_folder = input("Enter the path to the input folder: ").strip()
    output_processed_folder = input("Enter the path to the preferred txt data extraction: ").strip()
    ngram_output_file = input("Enter the path to the n-gram output CSV file: ").strip()
    ngram_type = input("Enter n-gram type (bigram or trigram): ").strip().lower()

    process_and_save_text(input_folder, output_processed_folder, ngram_output_file, ngram_type)
