# ngram_analysis
Ngram-based analytical idiolectal profiling

## Setting
Create a "source" folder. <br />
Create a "output" folder. <br />
Include your text files into the "source" folder to extract the bigrams or trigrams. This is your main data.<br />
<!-- Default files are generated with phi3:14b (93 participants x 7 genres). -->
Create a "corpus" folder. <br />
This is your preferred corpus for comparison.

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

## Extracting XML to TXT ##
Extracting XML-formatted corpus(es) to .txt files for comparison.
```[bash]
python3 xmlprocess.py
```
After pasting in the path of "/corpus/", the output will be produced as "/corpus_processed/"

## Run
```[bash]
python3 main.py
```

## Tips
For ease of reading CSV files, consider installing [Rainbow CSV](https://marketplace.visualstudio.com/items?itemName=mechatroner.rainbow-csv).<br />
``Ctrl + P`` in VsCode to launch Quick Open and paste:<br />
```ext install mechatroner.rainbow-csv``` 
