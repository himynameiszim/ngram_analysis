import os
import pandas as pd

def main():
    # Define paths
    input_folder = input("Enter the path to contestants' data: ")  # Update with your folder path
    output_folder = input("Enter the path to your output folder: ")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    input_file = input("Enter the path to corpus' data:  ")  # The provided file with ngrams and zscores

    # Load the ngram and zscore data
    bnc_baby_data = pd.read_csv(input_file)
    # Normalize ngrams in the input file to match the format "('word1', 'word2')"
    bnc_baby_data['ngram'] = bnc_baby_data['ngram'].apply(lambda x: str(tuple(x.split())))
    ngram_to_zscore = dict(zip(bnc_baby_data['ngram'], bnc_baby_data['z_score']))

    # Iterate over all CSV files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('z_score.csv'):
            file_path = os.path.join(input_folder, filename)
            
            # Load the current CSV file
            data = pd.read_csv(file_path)
            
            # Add a new column for z-scores
            # Ensure the ngrams are stripped of extraneous quotes
            data['ngram'] = data['ngram'].apply(lambda x: str(eval(x)))
            data['z-score_BNC_baby'] = data['ngram'].map(ngram_to_zscore).fillna(0)
            
            # Save the updated file to the output folder
            output_file_path = os.path.join(output_folder, filename)
            data.to_csv(output_file_path, index=False)

    print("All files updated and saved to the output folder successfully!")

main()