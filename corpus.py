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
from typing import Dict
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

# Define genre_list and order that ngram_list will be sorted
genre_list = ["interview", "pd1", "pd2", "tm-family", "tm-friends", "vm1", "vm2"]
order = ["ngram", "total_count", 
        "interview_count", "pd1_count", "pd2_count", "tm-family_count", "tm-friends_count", "vm1_count", "vm2_count", 
        "interview_freq", "pd1_freq", "pd2_freq", "tm-family_freq", "tm-friends_freq", "vm1_freq", "vm2_freq"]

def calculate_zscores_ngram_individual_totalcount(input_dir):

    os.makedirs(input_dir, exist_ok=True)

    # Read all input files
    suffix = "_zscores"
    # csv_files = sorted([file for file in os.listdir(input_dir) if file.endswith("_bigrams.csv")])
    csv_files = sorted([file for file in os.listdir(input_dir) if file.endswith("_trigrams.csv")])
    bigram_data = {file: pd.read_csv(os.path.join(input_dir, file)) for file in csv_files}
    total_num_of_participants = len(csv_files)

    '''
    Extract the first column (the ngram) and the second column (the "total_count") of each ngram extracted list
    total_count is the number of instances of that ngram used by the participant (in all 7 files/genres).
    '''
    for file, df in bigram_data.items():
        output_file = os.path.join(input_dir, f"{os.path.splitext(file)[0]}{suffix}.csv")
        df.iloc[:, :2].to_csv(output_file, index=False)

    '''
    Create a nested dictionary to store a list of appearance of the ngrams based on the genre
    e.g: the dictionary sigma_list[ngramX][genreY] = [x_0, x_1, x_2, ....] shows that the ngramX in genreY is used:
        x_i times by some participants
    this is done by simply iterating through all of the ngrams in the ngram extracted list and append the "{genre}_count" of that ngram into the above mentioned list.
    the genre/container name are defined in genre_list above.
    '''
    sigma_list = defaultdict(lambda: defaultdict(list)) # {ngram: {genre: [counts]}}
    for file, df in bigram_data.items():
        # for debug
        print(f"processing {file}")
        for genre in genre_list:
            count_col = f"{genre}_count"
            for _, row in df.iterrows():
                ngram = row["ngram"]
                sigma_list[ngram][genre].append(row[count_col])

    '''
    For ngrams that are not used by all participant, keep appending 0 into sigma_list[ngramX][genreY], which means its being used 0 time.
    And then, calculate arithmetic mean (\mu) and population standard deviation of EACH ngram wrt EACH container
    ''' 
    genre_stats = {}
    for ngram in sigma_list:
        print(f'Calculating mu and sd for "{ngram}"')
        genre_stats[ngram] = {}
        for genre in genre_list:
            while len(sigma_list[ngram][genre]) < total_num_of_participants:
                sigma_list[ngram][genre].append(0)
            genre_stats[ngram][genre] = {
            "mu": sum(sigma_list[ngram][genre]) / total_num_of_participants,
            "sd": stat.pstdev(sigma_list[ngram][genre])
            }

    # Calculate the zscore based on the x, mu and sd; then, attach them as a new column in the zscore file
    zscores_files = sorted([filez for filez in os.listdir(input_dir) if filez.endswith(f"_zscores.csv")])            
    zscore_data = {filez: pd.read_csv(os.path.join(input_dir, filez)) for filez in zscores_files}
    
    for filez, dfz in zscore_data.items():
        print(f"----------Calculating zscore for file {filez}----------")
        for genre in genre_list:
            zscore_col = f"{genre}_zscore"
            dfz[zscore_col] = 0.0
            mu_col = f"{genre}_mu"
            sd_col = f"{genre}_sd"

            # Caclulate the zscores for each row 
            for idx, row in dfz.iterrows():
                ngram = row["ngram"]
                sd = genre_stats[ngram][genre]["sd"]
                mu = genre_stats[ngram][genre]["mu"]

                dfz.at[idx, mu_col] = mu
                dfz.at[idx, sd_col] = sd

                if sd == 0.0:
                    continue
                
                total_count = int(dfz.loc[dfz["ngram"] == ngram, "total_count"])
                zscore = (total_count - mu) / sd
                dfz.at[idx, zscore_col] = zscore
        dfz.to_csv(os.path.join(input_dir, filez), index=False)

    print("Done!")

def calculate_zscores_ngram_individual_genrecount(input_dir):
    
    os.makedirs(input_dir, exist_ok = True)

    # read all inp files
    suffix = "_zscores"
    # csv_files = sorted([file for file in os.listdir(input_dir) if file.endswith("_bigrams.csv")])
    csv_files = sorted([file for file in os.listdir(input_dir) if file.endswith("_trigrams.csv")])
    bigram_data = {file: pd.read_csv(os.path.join(input_dir, file)) for file in csv_files}
    total_num_of_participants = len(csv_files)
    sigma_list = defaultdict(lambda: defaultdict(list)) # {ngram: {genre: [counts]}}

    '''
    Step 1: Extract the all the genre count from all the ngram extracted list and append to the new zscores calculation files.
    {genre}_count is the number of instances of that ngram used by the participant in that particular genre.

    Step 2: Create a nested dictionary to store a list of appearance of the ngrams based on the genre
    e.g: the dictionary sigma_list[ngramX][genreY] = [x_0, x_1, x_2, ....] shows that the ngramX in genreY is used:
        x_i times by some participants
    this is done by simply iterating through all of the ngrams in the ngram extracted list and append the "{genre}_count"
    of that ngram into the above mentioned list.
    the genre/container name are defined in genre_list above.
    '''
    for file, df in bigram_data.items():
        print(f"processing {file}")
        output_file = os.path.join(input_dir, f"{os.path.splitext(file)[0]}{suffix}.csv")
        output_df = df.iloc[:, :1]

        for genre in genre_list:
            count_col = f"{genre}_count"
            output_df[count_col] = df[count_col]
            for _, row in df.iterrows():
                ngram = row["ngram"]
                sigma_list[ngram][genre].append(row[count_col])

        output_df.to_csv(output_file, index=False)

    '''
    Step 3: For ngrams that are not used by all participant, keep appending 0 into sigma_list[ngramX][genreY], which means its being used 0 time.
    And then, calculate arithmetic mean (\mu) and population standard deviation of EACH ngram wrt EACH container
    ''' 
    genre_stats = {}
    for ngram in sigma_list:
        print(f'Calculating mu and sd for "{ngram}"')
        genre_stats[ngram] = {}
        for genre in genre_list:
            while len(sigma_list[ngram][genre]) < total_num_of_participants:
                sigma_list[ngram][genre].append(0)
            genre_stats[ngram][genre] = {
            "mu": sum(sigma_list[ngram][genre]) / total_num_of_participants,
            "sd": stat.pstdev(sigma_list[ngram][genre])
            }

    # Calculate the zscore based on the x, mu and sd; then, attach them as a new column in the zscore file
    zscores_files = sorted([filez for filez in os.listdir(input_dir) if filez.endswith(f"_zscores.csv")])            
    zscore_data = {filez: pd.read_csv(os.path.join(input_dir, filez)) for filez in zscores_files}
    
    for filez, dfz in zscore_data.items():
        print(f"----------Calculating zscore for file {filez}----------")
        for genre in genre_list:
            zscore_col = f"{genre}_zscore"
            dfz[zscore_col] = 0.0
            mu_col = f"{genre}_mu"
            sd_col = f"{genre}_sd"

            # Caclulate the zscores for each row 
            for idx, row in dfz.iterrows():
                ngram = row["ngram"]
                sd = genre_stats[ngram][genre]["sd"]
                mu = genre_stats[ngram][genre]["mu"]

                dfz.at[idx, mu_col] = mu
                dfz.at[idx, sd_col] = sd

                if sd == 0.0:
                    continue
                
                genre_count = int(dfz.loc[dfz["ngram"] == ngram, f"{genre}_count"])
                zscore = (genre_count - mu) / sd
                dfz.at[idx, zscore_col] = zscore
        dfz.to_csv(os.path.join(input_dir, filez), index=False)

    print("Done!")

def calculate_ngram_frequencies(directory: str) -> dict:
    """
    Calculate d_freq_corpus, d_freq_cps, and d_sum_cps from the ngram extracted lists in the given directory.
    
    :var d_freq_corpus stores the total counts of each bigram across all subcorpora (e.g., "quite a" appears 5 times in total).
    :var d_freq_cps stores the counts per subcorpus (e.g., in subcorpus "001", "quite a" appears 3 times).
    :var d_sum_cps stores the total count of bigrams in each subcorpus (e.g., "001" has a total of 10 bigrams).
    :param directory: The directory containing the ngram extracted lists.
    :return: A tuple containing d_freq_corpus, d_freq_cps, and d_sum_cps.
    """
    
    # Initialize dictionaries
    d_freq_corpus = defaultdict(int)  # Store total counts for each bigram across the corpus
    d_freq_cps = defaultdict(lambda: defaultdict(int))  # Store bigram counts per subcorpus
    d_sum_cps = defaultdict(int)  # Store total bigram count per subcorpus
    
    # ngram-choice
    ngram_choice = 'bigrams'
    # Iterate over all CSV files in the given directory
    for filename in os.listdir(directory):
        if filename.endswith(f"{ngram_choice}.csv"):
            subcorpus_id = filename.split("_")[0]  # Extract subcorpus ID from filename
            file_path = os.path.join(directory, filename)
            
            # Read bigrams
            with open(file_path, mode="r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                total_bigrams_subcorpus = 0  # Track the total count of bigrams for this subcorpus
                
                for row in reader:
                    bigram = row["ngram"]
                    count = int(row[f"{subcorpus_id}_count"])
                    
                    # Update d_freq_cps for the current subcorpus
                    d_freq_cps[subcorpus_id][bigram] += count
                    
                    # Update d_freq_corpus for the entire corpus
                    d_freq_corpus[bigram] += count
                    
                    # Update total bigram count for the subcorpus
                    total_bigrams_subcorpus +=1
                
                # Update d_sum_cps with the total bigram count for this subcorpus ? total words -> consider
                d_sum_cps[subcorpus_id] = total_bigrams_subcorpus
    
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

    d_dp = {}

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
        d_dp[bigram] = dp / (1 - smallest_cp) if (1 - smallest_cp) > 0 else 0

    return d_dp

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
        
        # Save the reordered DataFrame to the output directory
        df.to_csv(output_path, index=False)
        print(f"Sorted columns for {file} and saved to {output_path}")

if __name__ == "__main__":

    input_dir = input("Enter the path to the input directory: ").strip()

    preprocess_data(input_dir)

    output_dir = input("Enter the path to the output directory: ").strip()

    start_time = datetime.now()
    main(input_dir, output_dir)
    
    # calculate_zscores_ngram_individual_genrecount(output_dir)
    d_freq_corpus, d_freq_cps, d_sum_cps = calculate_ngram_frequencies(output_dir)
    
    # for debug
    # print("d_freq_corpus:", dict(d_freq_corpus))
    # print("d_freq_cps:", dict(d_freq_cps))
    # print("d_sum_cps:", dict(d_sum_cps))

    d_dp_norm = calculate_dpnorm(d_freq_corpus, d_freq_cps, d_sum_cps) #calculate dpnorm of each ngram

    # write to log file
    logfile = "log.txt"
    output_filename = logfile
    with open(output_filename, mode="w", encoding="utf-8") as f:
        f.write("ngram, DPnorm\n")
        for bigram, dp_value in sorted(d_dp_norm.items(), key=lambda x: x[1], reverse=True):  # Sort by DPnorm (highest first)
            f.write(f"{bigram},{dp_value:.6f}\n")

    end_time = datetime.now()
    duration = end_time - start_time

    print(f"Program duration: {duration}\n Done!")
