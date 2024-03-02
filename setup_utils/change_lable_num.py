import os

folder_path = '/path/to/your/folder'  # Replace with the actual path to your folder

def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    updated_lines = [line.replace('0', '3', 1) if line.startswith('0') else line for line in lines]

    with open(file_path, 'w') as file:
        file.writelines(updated_lines)

def process_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            process_file(file_path)
            # print(f"Processed: {filename}")

if __name__ == "__main__":
    # Call the function to process files in the specified folder
    process_files_in_folder(folder_path)
