"""Small utility helpers for file IO and task identifiers."""

import hashlib
import json
import os


def load_json_data(file_path):
    """Load and return JSON from a file path."""
    with open(file_path) as file:
        return json.load(file)


def get_init_files(directory_path):
    """Return a list of .json files within a directory (non-recursive)."""
    json_file_paths = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            json_file_paths.append(file_path)
    return json_file_paths


def load_specification(file_path):
    """Read a specification text file relative to the current working directory."""
    try:
        current_directory = os.getcwd()
        spec_path = current_directory + file_path
        with open(spec_path) as file:
            return file.read()
    except FileNotFoundError:
        return "File not found."


def get_task_identifier(sender, content):
    """Create a stable task ID using the sender and a short hash of content."""
    content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
    task_identifier = f"{sender}_{content_hash}"

    return task_identifier
