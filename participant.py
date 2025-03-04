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
import csv
import statistics as stat
from datetime import datetime
import warnings
from typing import Dict, List, Tuple
warnings.filterwarnings('ignore')

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

def get_word_count(file_path):
     with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            return len(text.split())  # Split by whitespace and count words

def count_words_in_files(directory, substring):
    """Count the total words in all text files in a directory with a certain substring in the filename."""
    total_word_count = 0

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt") and substring in filename:
            file_path = os.path.join(directory, filename)
            total_word_count += get_word_count(file_path)

    return total_word_count

# Function to remove punctuation from text
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# Parse filenames
def parse_filename(filename):
    parts = filename.split('_')
    # contestant_id = parts[0]
    # gender = parts[1]
    # genre = parts[2] 
    # word_count = int(parts[3].replace('.txt', ''))
    return parts[0], parts[1], parts[2], int(parts[3].replace('.txt', ''))

# Read and group text files
def read_text_files(directory):
    contestant_files = defaultdict(list)
    for file in os.listdir(directory):
        if file.endswith('.txt'):
            filepath = os.path.join(directory, file)
            contestant_id, gender, genre, word_count = parse_filename(file)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read().lower().replace('\n', ' ')
                text = remove_punctuation(text)  # Remove punctuation from the text
            # contestant_files[contestant_id].append((gender, container, word_count, text))
            contestant_files[contestant_id].append((genre, text))
    return contestant_files

# Extract n-grams
def extract_ngrams(text, ngram_type):
    tokens = [word for word in nltk.word_tokenize(text) if word.isalnum()]
    return list(bigrams(tokens)) if ngram_type == 2 else list(trigrams(tokens))

# Generate CSV files for n-grams
def generate_csv_files(contestant_id, contestant_data, n, output_dir):
    containers = [data[0] for data in contestant_data]
    total_ngrams = Counter()
    container_ngrams = defaultdict(Counter)

    # Calculate n-grams and counts
    for container, text in contestant_data:
        ngrams = extract_ngrams(text, n)
        total_ngrams.update(ngrams)
        container_ngrams[container].update(ngrams)
    
    # Calculate statistics
    rows = []
    for ngram, count in total_ngrams.items():
        row = {"ngram": " ".join(ngram), f'{contestant_id}_count': count}

        # Add container-specific data
        for container in containers:
            container_count = container_ngrams[container][ngram]
            row[f'{container}_count'] = container_count
        rows.append(row)

    # Prepare columns
    columns = ['ngram', f'{contestant_id}_count'] + \
              [f'{container}_count' for container in containers]
    output_file = os.path.join(output_dir, f'{contestant_id}_{"bigrams" if n == 2 else "trigrams"}.csv')
    pd.DataFrame(rows, columns=columns).to_csv(output_file, index=False)
    print(f"N-gram CSV file saved: {output_file}")

def calculate_ngram_frequencies(file_path: str) -> Tuple[Dict, Dict, Dict]:
    """
    Calculate d_freq_corpus, d_freq_cps, and d_sum_cps from the given ngram extracted CSV file.
    
    :var d_freq_corpus stores the total counts of each n-gram across all subcorpora (e.g., "quite a" appears 5 times in total).
    :var d_freq_cps stores the counts per subcorpus (e.g., in subcorpus "001", "quite a" appears 3 times).
    :var d_sum_cps stores the total count of n-grams in each subcorpus (e.g., "001" has a total of 10 n-grams).
    
    :param file_path: The exact path to the n-gram extracted CSV file.
    :param ngram_choice: 2 for bigrams, 3 for trigrams.
    
    :return: A tuple containing (d_freq_corpus, d_freq_cps, d_sum_cps).
    """
    
    # Initialize dictionaries
    d_freq_corpus = defaultdict(int)  # Store total counts for each n-gram across the corpus
    d_freq_cps = defaultdict(lambda: defaultdict(int))  # Store n-gram counts per subcorpus
    d_sum_cps = defaultdict(int)  # Store total n-gram count per subcorpus

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Extract subcorpus ID from filename (assuming format "001_bigrams.csv")
    subcorpus_id = os.path.basename(file_path).split("_")[0]  

    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        total_ngrams_subcorpus = 0  # Track total count of n-grams for this subcorpus
        
        for row in reader:
            ngram = row["ngram"]
            count = int(row[f"{subcorpus_id}_count"])
            
            # Update dictionaries
            d_freq_cps[subcorpus_id][ngram] += count
            d_freq_corpus[ngram] += count
            total_ngrams_subcorpus += count
        
        # Update d_sum_cps with total n-gram count for this subcorpus
        d_sum_cps[subcorpus_id] = total_ngrams_subcorpus

    return d_freq_corpus, d_freq_cps, d_sum_cps


def calculate_dpnorm(d_freq_corpus: Dict, d_freq_cps: Dict, d_sum_cps: Dict) -> Dict:
    """Compute DPnorm (dispersion normalized) for bigrams across subcorpora.
    
    :param d_freq_corpus: Dictionary with total bigram counts in the entire corpus.
    :param d_freq_cps: Dictionary with bigram frequencies per subcorpus.
    :param d_sum_cps: Dictionary with total bigram counts per subcorpus.
    :return: Dictionary with DPnorm values for each bigram.
    """

    # If there's only one subcorpus, all DP values are zero
    if len(d_sum_cps) == 1:
        return {bigram: 0 for bigram in d_freq_corpus}

    # Compute total bigrams across the entire corpus
    total_bigrams_corpus = sum(d_sum_cps.values())  
    smallest_cp = min(d_sum_cps.values()) / total_bigrams_corpus if total_bigrams_corpus > 0 else 0

    d_dp_norm = {}

    # Compute DP for each bigram
    for bigram in d_freq_corpus:
        dp = 0  # Initialize DP value
        
        # Total frequency of this bigram in the entire corpus
        freq_corpus = d_freq_corpus[bigram]

        # Iterate over each subcorpus
        for subcorpus, total_bigrams_subcorpus in d_sum_cps.items():
            # Expected frequency (E)
            expected = total_bigrams_subcorpus / total_bigrams_corpus if total_bigrams_corpus > 0 else 0

            # Observed frequency (O)
            freq_part = d_freq_cps.get(subcorpus, {}).get(bigram, 0)
            observed = freq_part / freq_corpus if freq_corpus > 0 else 0

            # Compute absolute difference
            abs_diff = abs(expected - observed)
            dp += abs_diff * 0.5

        # Normalize DPnorm
        d_dp_norm[bigram] = dp / (1 - smallest_cp) if (1 - smallest_cp) > 0 else 0

    return d_dp_norm

def d_freq_abs_adj(d_freq_corpus: Dict, d_dp_norm: Dict) -> Dict:
    '''Add adjusted frequencies to frequency dictionary (per ngram)
    :param d_freq_corpus: frequency dictionaru of the ngram across the entire corpus.
    :param d_dp_norm: dictionary containing the dpNorm value of each ngram.

    :var d_adj_freq: Dictionary storing adjusted frequencies for each ngram.
    :var raw_freq: Raw count of the ngram in the corpus before adjustment.
    :var adjusted_freq: Adjusted frequency using DPnorm formula.
    :var abs_freq_lapl: Laplace-smoothed absolute frequency of ngram.
    :var adj_freq_lapl: Laplace-smoothed adjusted frequency of ngram.

    :return: the frequency dictionary adjusted based on dpNorm     
    '''
    d_abs_adj = {}

    for ngram in d_freq_corpus:
        d_ngram = {}
        dp_score = d_dp_norm[ngram]
        raw_freq = d_freq_corpus[ngram]

        #adj frequency based on dpnorm
        adj_freq = raw_freq * ( 1 - dp_score )
        abs_freq_lapl = raw_freq + 1 #laplace smoothing
        adj_freq_lapl = adj_freq + 1 #laplace smooth also the adjusted frequency

        d_ngram["DP"] = dp_score
        d_ngram["abs_freq"] = raw_freq
        d_ngram["adj_freq"] = adj_freq
        d_ngram["abs_freq_lapl"] = abs_freq_lapl
        d_ngram["adj_freq_lapl"] = adj_freq_lapl

        d_abs_adj[ngram] = d_ngram

    return d_abs_adj

def sum_abs_adj(d_abs_adj: Dict) -> Dict:
    """Add adjusted frequencies to frequency dictionary (total).
    :param d_abs_adj: frequency dictionary enriched with adjusted frequency values.
    :param desired_pos: tuple of UD tags which should be taken into account in the keyness calculations.
    :return: the frequency dictionary enriched with adjusted frequency values.
    """
    # Initialize dictionary to store sum of adjusted frequencies
    d_sum_abs_adj = {"all": {"abs_freq": 0, "adj_freq": 0, "abs_freq_lapl": 0, "adj_freq_lapl": 0, "unique": 0}}

    # Iterate over all the items in the adjusted frequency dictionary
    for tup in d_abs_adj:
        # Add the frequencies for "all"
        d_sum_abs_adj["all"]["abs_freq"] += d_abs_adj[tup]["abs_freq"]
        d_sum_abs_adj["all"]["adj_freq"] += d_abs_adj[tup]["adj_freq"]
        d_sum_abs_adj["all"]["abs_freq_lapl"] += d_abs_adj[tup]["abs_freq_lapl"]
        d_sum_abs_adj["all"]["adj_freq_lapl"] += d_abs_adj[tup]["adj_freq_lapl"]
        d_sum_abs_adj["all"]["unique"] += 1

    return d_sum_abs_adj

def main():
    print("\n------Step 1: Ngram extraction and count------\n")

    input_dir = input("Enter the path to the input directory: ").strip()
    output_dir = input("Enter the path to the output directory: ").strip()
    preprocess_data(input_dir)
    
    # Remove the output directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
       
    os.makedirs(output_dir, exist_ok=True)

    # Ask the user to choose between bigrams or trigrams
    ngram_choice = input("Choose n-gram type (2 for bigram, 3 for trigram): ").strip()
    while ngram_choice not in ['2', '3']:
        ngram_choice = input("Invalid choice. Please enter 2 for bigram or 3 for trigram: ").strip()
    ngram_choice = int(ngram_choice)
    if ngram_choice == 2:
        suffix = "_bigrams.csv"
    elif ngram_choice == 3:
        suffix = "_trigrams.csv"

    # For contestant_data
    contestant_files = read_text_files(input_dir)

    for contestant_id, contestant_data in contestant_files.items():
        # Generate the selected n-gram CSV file
        generate_csv_files(contestant_id, contestant_data, ngram_choice, output_dir)

    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
    for file in csv_files:
        input_path = os.path.join(output_dir, file)
        output_path = os.path.join(output_dir, file)
        
        # Load the CSV file
        df = pd.read_csv(input_path)
        
        # Save to the output directory
        df.to_csv(output_path, index=False)
        print(f"{file} saved to {output_path}")

    print("\n------Step 2: Keyness calculation------\n")

    sc_dir = input("Path to the study corpus directory(ngram extracted list): ").strip()
    rc_dir = input("Path to the reference corpus directory(ngram extracted list): ").strip()
    rc_file = os.path.join(rc_dir, f"93{suffix}")

    # keyn_metric = input("Your choice of keyness metric (%DIFF, Ratio, OddsRatio, LogRatio, DiffCoefficient): ").strip()

    #get ngram frequencies dict of rc
    d_freq_corpus_rc, d_freq_cps_rc, d_sum_cps_rc = calculate_ngram_frequencies(rc_file)
    #calculate dpnorm of each ngram
    d_dp_norm_rc = calculate_dpnorm(d_freq_corpus_rc, d_freq_cps_rc, d_sum_cps_rc) 
    #compute adjusted frequencies
    d_abs_adj_rc = d_freq_abs_adj(d_freq_corpus_rc, d_dp_norm_rc) 
    #compute total adjusted frequencies
    d_sum_abs_adj_rc = sum_abs_adj(d_abs_adj_rc)

    approx_num = 0.1
    freq_type = "adj_freq"

    rc_file = os.path.join(rc_dir, f"93{suffix}")
    # rc_data = pd.read_csv(rc_file, usecols=["ngram", "93_count"])
    # d_rc = dict(zip(rc_data["ngram"], rc_data["93_count"]))

    for i in range(1, 94):
        sc_name = f"{i:03d}"
        sc_file = os.path.join(sc_dir, f"{sc_name}{suffix}")

        #get ngram frequencies dict of rc
        d_freq_corpus_sc, d_freq_cps_sc, d_sum_cps_sc = calculate_ngram_frequencies(sc_file)
        #calculate dpnorm of each ngram
        d_dp_norm_sc = calculate_dpnorm(d_freq_corpus_sc, d_freq_cps_sc, d_sum_cps_sc) 
        #compute adjusted frequencies
        d_abs_adj_sc = d_freq_abs_adj(d_freq_corpus_sc, d_dp_norm_sc) 
        #compute total adjusted frequencies
        d_sum_abs_adj_sc = sum_abs_adj(d_abs_adj_sc)

        sc_data = pd.read_csv(os.path.join(sc_dir, f"{sc_name}{suffix}"))
        print(f"-----Calculating keyness for {sc_file}-----")

        col_to_del = ["pd2_count", "pd1_count", "vm2_count", "interview_count", "tm-family_count", "tm-friends_count", "vm1_count"]
        sc_data.drop(columns=col_to_del, inplace=True, errors='ignore')

        for index, row in sc_data.iterrows():
            ngram = row['ngram']
            freq_sc = d_abs_adj_sc[ngram][freq_type]
            freq_rc = d_abs_adj_rc[ngram][freq_type]
            # rc_count = int(d_rc.get(ngram, approx_num))

            sum_sc = d_sum_abs_adj_sc["all"][freq_type]
            sum_rc = d_sum_abs_adj_rc["all"][freq_type]

            norm_freq_1000_sc = freq_sc / sum_sc * 1000
            norm_freq_1000_rc = freq_rc / sum_rc * 1000

            # if keyn_metric == "%DIFF":
            keyn_score_sc = ((norm_freq_1000_sc - norm_freq_1000_rc) * 100) / norm_freq_1000_rc
            keyn_score_rc = ((norm_freq_1000_rc - norm_freq_1000_sc) * 100) / norm_freq_1000_sc

            sc_data.at[index, "keyness_%DIFF_93"] = keyn_score_sc
            # elif keyn_metric == "Ratio":
            keyn_score_sc = norm_freq_1000_sc / norm_freq_1000_rc
            keyn_score_rc = norm_freq_1000_rc / norm_freq_1000_sc

            sc_data.at[index, "keyness_Ratio_93"] = keyn_score_sc
            # elif keyn_metric == "OddsRatio":
            keyn_score_sc = (freq_sc / (sum_sc - freq_sc)) / (freq_rc / (sum_rc - freq_rc))
            keyn_score_rc = (freq_rc / (sum_rc - freq_rc)) / (freq_sc / (sum_sc - freq_sc))

            sc_data.at[index, "keyness_OddsRatio_93"] = keyn_score_sc
            # elif keyn_metric == "LogRatio":
            keyn_score_sc = np.log2(norm_freq_1000_sc / norm_freq_1000_rc)
            keyn_score_rc = np.log2(norm_freq_1000_rc / norm_freq_1000_sc)

            sc_data.at[index, "keyness_LogRatio_93"] = keyn_score_sc
            # elif keyn_metric == "DiffCoefficient":
            keyn_score_sc = (norm_freq_1000_sc - norm_freq_1000_rc) / (norm_freq_1000_sc + norm_freq_1000_rc)
            keyn_score_rc = (norm_freq_1000_rc - norm_freq_1000_sc) / (norm_freq_1000_rc + norm_freq_1000_sc)
            
            sc_data.at[index, "keyness_DiffCoefficient_93"] = keyn_score_sc
            # else:
            #     raise ValueError("`keyness metric` is not correctly defined.")
            
        # Save updated study corpus file
        sc_data.to_csv(sc_file, index=False)

if __name__ == "__main__":
    main()
