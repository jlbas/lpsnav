import sys

def print_to_file(file, message):
    original_stdout = sys.stdout
    with open(file, 'w') as outfile:
        sys.stdout = outfile
        print(message)
        sys.stdout = original_stdout
