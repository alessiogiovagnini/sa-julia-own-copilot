from tree_sitter import Parser, Language
import tree_sitter_julia
import sys
import csv
import os

# Load the Julia language
JULIA_LANGUAGE = Language(tree_sitter_julia.language())
parser = Parser(JULIA_LANGUAGE)

def extract_functions(node, code, functions):
    if node.type == "function_definition":
        # Extract the full function text
        function_text = code[node.start_byte:node.end_byte].decode("utf8")
        functions.append(function_text)
        
    # Recursively visit children nodes
    for child in node.children:
        extract_functions(child, code, functions)

def save_to_csv(repo_name, file_path, output_file, functions):

    # Check if the file exists to set write mode
    write_mode = "a" if os.path.exists(output_file) else "w"

    # Write to CSV
    with open(output_file, write_mode, newline='', encoding="utf8") as file:
        writer = csv.writer(file)
        
        # Write header if file is being created
        if write_mode == "w":
            writer.writerow(["repository_name", "file_path", "function"])

        # Write each function to CSV
        for function_text in functions:
            writer.writerow([repo_name, file_path, function_text])

def main(repo_name, file_path, output_file):
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
        main(sys.argv[1], sys.argv[2], sys.argv[3])
