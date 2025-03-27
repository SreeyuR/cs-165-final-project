import os

def replace_in_file(filepath, old_str, new_str):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        print(f"Skipping non-text file: {filepath}")
        return

    if old_str in content:
        new_content = content.replace(old_str, new_str)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated: {filepath}")

def replace_in_project(project_dir, old_str, new_str):
    for root, dirs, files in os.walk(project_dir):
        for file in files:
            filepath = os.path.join(root, file)
            replace_in_file(filepath, old_str, new_str)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Recursively replace strings in a project")
    parser.add_argument("project_dir", help="Path to the project root directory")
    parser.add_argument("old_string", help="String to be replaced")
    parser.add_argument("new_string", help="New string to replace with")

    args = parser.parse_args()
    replace_in_project(args.project_dir, args.old_string, args.new_string)
