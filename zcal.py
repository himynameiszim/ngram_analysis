import os
import pandas as pd
import numpy as np
from collections import defaultdict
import statistics as stat
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Define genres and column order
genre_list = ["interview", "pd1", "pd2", "tm-family", "tm-friends", "vm1", "vm2"]
order = [ "ngram", "total_count", "normalized_freq", 
    "interview_count", "pd1_count", "pd2_count", "tm-family_count", "tm-friends_count", "vm1_count", "vm2_count",
    "interview_freq", "pd1_freq", "pd2_freq", "tm-family_freq", "tm-friends_freq", "vm1_freq", "vm2_freq"
]

def calculate_zscores_bigram_individual_genre(input_dir):

    os.makedirs(input_dir, exist_ok=True)

    # Read all input files
    suffix = "_zscores"
    csv_files = sorted([file for file in os.listdir(input_dir) if file.endswith("_bigrams.csv")])
    bigram_data = {file: pd.read_csv(os.path.join(input_dir, file)) for file in csv_files}
    total_num_of_participants = len(csv_files)

    # Extract first two columns of bigram files to get num of instances (x)
    for file, df in bigram_data.items():
        output_file = os.path.join(input_dir, f"{os.path.splitext(file)[0]}{suffix}.csv")
        df.iloc[:, :2].to_csv(output_file, index=False)

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
    For ngram that is not used by all users, assume that the usage is 0 times
    And then, Calculate arithmetic mean (\mu) and population standard deviation of EACH ngram wrt EACH container
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
        print(f"Calculating zscore for file {filez}")
        for genre in genre_list:
            zscore_col = f"{genre}_zscore"
            dfz[zscore_col] = 0.0

            # Caclulate the zscores for each row 
            for idx, row in dfz.iterrows():
                ngram = row["ngram"]
                sd = genre_stats[ngram][genre]["sd"]
                # print(genre_stats[ngram][genre]["sd"])
                if sd == 0.0:
                    continue
                mu = genre_stats[ngram][genre]["mu"]
                total_count = int(dfz.loc[dfz["ngram"] == ngram, "total_count"])
                zscore = (total_count - mu) / sd
                # print(f'zscore of "{ngram}" is calculated by ( {total_count} - {genre_stats[ngram][genre]["mu"]} ) / {genre_stats[ngram][genre]["sd"]} ')
                dfz.at[idx, zscore_col] = zscore
        dfz.to_csv(os.path.join(input_dir, filez), index=False)

    # print("Done!")

input_dir = "/home/jimmy/Videos/test_ngram_analysis/participant_output/bigrams"
start_time = datetime.now()
calculate_zscores_bigram_individual_genre(input_dir)
end_time = datetime.now()
duration = end_time - start_time
print(f"Program duration: {duration}")