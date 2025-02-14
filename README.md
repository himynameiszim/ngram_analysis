# ngram_analysis
Ngram-based analytical idiolectal profiling

## Setting
Create a `contestant_source/` folder. Put your participants' data here. <br />
Create a `contestant_output` folder. N-gram extraction for contestant directory. <br />
Create a `corpus_source/` folder. Put your preferred corpus for comparison. <br />
Create a `corpus_output/` folder. N-gram extraction for corpus directory.


## Description ##
Samples of language from 7 different genres were collected from `n` participants. The genres include both spoken and written English. The spoken genres were transcribed and saved in text files in written form.
The primary purpose of this is to identify the bigrams and trigrams that are shared across all genres. The text file for one genre is called a container. Punctuation will be ignored.<br />
Several corpus of choice will be chosen for comparison between the n-grams' distribution.

## First time set up
Create an environment with Python 3.9 (Stable Version)
```[bash]
conda create -n <environment-name> python=3.9
conda activate <environment-name>
pip3 install -r requirements.txt
```

## Process data ##
The processing pipeline starts with ngram extraction and zscore calculation data from participants' data, then the same procedure is applied to reference corpus'. Finally, zscore comparison between the corpuss and participants' data.

### for participants ###
```[bash]
python3 participant.py
```
After pasting in the path of `corpus_source/` and `corpus_output/`, the n-gram extraction and z-score calculation will be extracted to this directory.

### for corpus ###

```[bash]
python3 corpus.py
```
After pasting in the path of `input_folder/` and `output_processed_folder/`, the .txt data extraction will be saved in `output_processed_folder`. Input in the `ngram_output_file` and `ngram_type`, the n-gram extraction and z-score calculation will be extracted to the `output_file`.

## Run ##
```[bash]
python3 main.py
```
Paste in your input participants' data directory `input_folder`. (the data that was previously processed) <br />
Paste in your input file
Paste in your output 
 input_folder = input("Enter the path to contestants' data: ")  # Update with your folder path
    output_folder = input("Enter the path to your output folder: ")
    input_file = input("Enter the path to reference corpus' data:  ")  # The provided file with ngrams and zscores

## Tips
For ease of reading CSV files, consider installing [Rainbow CSV](https://marketplace.visualstudio.com/items?itemName=mechatroner.rainbow-csv).<br />
``Ctrl + P`` in VsCode to launch Quick Open and paste:<br />
```ext install mechatroner.rainbow-csv``` 
