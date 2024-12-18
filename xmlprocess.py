import os
import xml.etree.ElementTree as ET

# Function to extract text recursively and write to a file
def extract_text_to_file(element, file):
    for child in element:
        if child.text and child.text.strip():
            file.write(f"{child.tag}: {child.text.strip()}\n")
        extract_text_to_file(child, file)

# Function to process a file
def process_file(input_file, output_file):
    with open(input_file, "r") as infile:
        lines = infile.readlines()

    # Extract text after ':' and concatenate
    result = ""
    for line in lines:
        if ':' in line:
            _, text = line.split(':', 1)
            result += text.strip() + " "

    # Write concatenated text to output file
    with open(output_file, "w") as outfile:
        outfile.write(result.strip())

# Function to process the entire folder structure
def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(input_folder):
        # Recreate directory structure in output folder
        relative_path = os.path.relpath(root, input_folder)
        current_output_folder = os.path.join(output_folder, relative_path)
        if not os.path.exists(current_output_folder):
            os.makedirs(current_output_folder)

        for file_name in files:
            if file_name.endswith(".xml"):
                input_file_path = os.path.join(root, file_name)
                temp_output_file = os.path.join(current_output_folder, f"{file_name}.txt")
                final_output_file = os.path.join(current_output_folder, file_name)

                # Extract and process text
                with open(temp_output_file, "w") as temp_file:
                    tree = ET.parse(input_file_path)
                    root_element = tree.getroot()
                    extract_text_to_file(root_element, temp_file)

                process_file(temp_output_file, final_output_file + "_processed.txt")

                # Remove the temporary file
                os.remove(temp_output_file)

# Main execution
if __name__ == "__main__":
    input_folder = input("Enter the path to the input directory: ").strip()
    # input_folder = "BNC_baby_4mill/"
    output_folder = input_folder.rstrip("/") + "_processed/"

    process_folder(input_folder, output_folder)
    print(f"Processed files are saved in '{output_folder}'.")
