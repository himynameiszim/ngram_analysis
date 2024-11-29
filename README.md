# ngram_analysis
Simple ngram_analytic Python Jupyter Notebook

## Setting

Include your text files into the "source" folder to extract the bigrams or trigrams.<br />
<!-- Include your sample bigrams to be compared into the folder "test_inp".<br /> -->
Default files are generated with phi3:14b (100 participants x 7 genres).

## Description ##
Samples of language from 7 different genres were collected from `n` participants. The genres include both spoken and written English. The spoken genres were transcribed and saved in text files in written form.
The primary purpose of this is to identify the bigrams and trigrams that are shared across all genres. The text file for one genre is called a container. Punctuation will be ignored.

## First time set up

Create an environment with Python 3.9 (Stable Version)
```[bash]
conda create -n <environment-name> python=3.9
conda activate <environment-name>
pip3 install -r requirements.txt
```

## Run

```[bash]
python3 main.py
```

## Tips

For ease of reading CSV files, consider installing [Rainbow CSV](https://marketplace.visualstudio.com/items?itemName=mechatroner.rainbow-csv).<br />
``Ctrl + P`` in VsCode to launch Quick Open and paste:<br />
```ext install mechatroner.rainbow-csv``` 
