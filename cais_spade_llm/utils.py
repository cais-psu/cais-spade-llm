import json
import os
import time, hashlib

def load_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Get the paths of all JSON files in the specified directory.
def get_init_files(directory_path):
    json_file_paths = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            json_file_paths.append(file_path)
    return json_file_paths

def load_specification(file_path):
    try:
        current_directory = os.getcwd()
        spec_path = current_directory + file_path
        with open(spec_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return "File not found."

def get_task_identifier(sender, content):
    content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
    task_identifier = f"{sender}_{content_hash}"

    return task_identifier 
