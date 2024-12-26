import os
import xml.etree.ElementTree as ET

def remove_metadata_and_extract_text(file_path):
    """
    Remove metadata from the XML file and extract text content.

    :param file_path: Path to the XML file.
    :return: Extracted text content without metadata.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Remove the <header> element if present
        header = root.find('header')
        if header is not None:
            root.remove(header)

        # Extract text from all relevant elements in <body>
        body = root.find('body')
        if body is None:
            print(f"No <body> found in {file_path}")
            return ""

        text_content = []
        for elem in body.iter():
            if elem.text:
                text_content.append(elem.text.strip())

        return ' '.join(text_content)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return ""

def process_and_save_text(folder_path, output_folder):
    """
    Process XML files in a folder by removing metadata, extracting text, 
    and saving it into separate text files.

    :param folder_path: Path to the folder containing XML files.
    :param output_folder: Path to the folder where text files will be saved.
    """
    try:
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Iterate over all files in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            # Check if it is an XML file
            if os.path.isfile(file_path) and file_name.endswith('.xml'):
                text = remove_metadata_and_extract_text(file_path)

                # Save the processed text to a corresponding text file
                output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.txt")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(text)

        print(f"Text successfully processed and saved to {output_folder}")

    except Exception as e:
        print(f"Error processing folder {folder_path}: {e}")

if __name__ == "__main__":
    input_folder = input("Enter the path to untagged corpus data: ").strip()
    output_folder = input("Enter the path to the folder where text files will be saved: ").strip()

    process_and_save_text(input_folder, output_folder)
