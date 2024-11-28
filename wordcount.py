import os
import string

def read_file(filepath):
    """
    Reads a single text file and returns its content.
    
    Args:
        filepath (str): Path to the text file.
    
    Returns:
        str: Content of the file as a single string.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

def preprocess_text(text):
    """
    Preprocesses text by removing punctuation and converting it to lowercase.
    
    Args:
        text (str): The input text to preprocess.
    
    Returns:
        str: Cleaned text with punctuation removed and in lowercase.
    """
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator).lower()

def count_words_in_file(filepath):
    """
    Counts the total number of words in a file after preprocessing.
    
    Args:
        filepath (str): Path to the text file.
    
    Returns:
        int: Total word count in the file.
    """
    text = read_file(filepath)
    cleaned_text = preprocess_text(text)
    words = cleaned_text.split()
    return len(words)

def count_words_in_folder(folder_path):
    """
    Counts the total number of words in each file in a folder.
    
    Args:
        folder_path (str): Path to the folder containing text files.
    
    Returns:
        dict: A dictionary with filenames as keys and their word counts as values.
    """
    word_counts = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            word_count = count_words_in_file(file_path)
            word_counts[filename] = word_count
    return word_counts

def rename_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            # count the words in the file
            word_count = count_words_in_file(file_path)

            # make new file name 
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_{word_count}{ext}"
            new_file_path = os.path.join(folder_path, new_filename)

            # rename the file 
            os.rename(file_path, new_file_path)
        print(f"Renamed '{filename}' to '{new_filename}' ")


if __name__ == "__main__":
    # Set the folder path containing the text files
    folder_path = "/mnt/fa80f336-3342-4d78-8bfd-a43e434a2cda/ngram_analysis/ngram_analysis/source"
    
    # Count words in each file
    word_counts = count_words_in_folder(folder_path)
    
    # Display results
    print("Word Count Per File:")
    for filename, count in word_counts.items():
        print(f"{filename}: {count} words")

    # rename the files 
    rename_files(folder_path)
