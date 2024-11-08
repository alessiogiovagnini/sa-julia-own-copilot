from pathlib import Path

from tree_sitter import Parser, Language
import tree_sitter_julia
import sys
import csv
import os
from tqdm import tqdm

# Load the Julia language
JULIA_LANGUAGE = Language(tree_sitter_julia.language())
parser = Parser(JULIA_LANGUAGE)

i = 0


def extract_functions(node, code, functions):
    # print(node.type)

    if node.type == "function_definition":
        # Extract the full function text
        function_text = code[node.start_byte:node.end_byte].decode("utf8")
        
        signature, body = split_function(function_text)
        
        docstring = find_docstring_before_function(code, node.start_byte)
        
        functions.append((docstring, signature, body))

    # Recursively visit children nodes
    for child in node.children:
        extract_functions(child, code, functions)

def split_function(function_str):
    # Split the function string into lines
    lines = function_str.split('\n')
    
    # First line is the function signature
    signature = lines[0]
    
    # The rest of the function is the body (join remaining lines)
    body = '\n'.join(lines[1:])
    
    return signature, body

def find_docstring_before_function(code, start_byte):
    previous_text = function_text = code[0:start_byte].decode("utf8")

    if previous_text[-4: -1] == '"""':

        # we split after each triple double quotes to get the last docstring
        previous_text_subsections = previous_text.split('"""')
        return previous_text_subsections[-2]
    else:
        return None


def save_to_csv(repo_name: str, file_path: Path, output_file: Path, functions):
    # Check if the file exists to set write mode
    write_mode = "a" if os.path.exists(output_file) else "w"

    # Write to CSV
    with open(output_file, write_mode, newline='', encoding="utf8") as file:
        writer = csv.writer(file)

        # Write header if file is being created
        if write_mode == "w":
            writer.writerow(["repository_name", "file_path", "docstring", "function_signature", "function_body"])

        # Write each function to CSV
        
        print('Writing on file:')
        for function in tqdm(functions):
            writer.writerow([repo_name, file_path, function[0], function[1], function[2]])


def start_extraction(repo_name: str, file_path: Path, output_file: Path):
    # Read code from file
    with open(file_path, "r", encoding="utf8") as file:
        code = file.read()

    # Parse the code
    tree = parser.parse(bytes(code, "utf8"))

    # Extract functions
    functions = []
    extract_functions(tree.root_node, bytes(code, "utf8"), functions)

    # Save to CSV
    save_to_csv(repo_name, file_path, output_file, functions)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python julia_function_extractor.py <repository_name> <path_to_julia_file> <output_file>")
    else:
        start_extraction(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]))
