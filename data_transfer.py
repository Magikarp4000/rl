import os, json


def load(file_name):
    # Load json file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    load_file_path = f'{dir_path}\\{file_name}.json'
    try:
        to_load = json.load(open(load_file_path, 'r'))
        return to_load
    except FileNotFoundError:
        print("LOAD_ERROR: File doesn't exist!\n")

def save(file_name, to_save={}):
    # Save into json file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    save_file_path = f'{dir_path}\\{file_name}.json'
    open(save_file_path, 'w').write(json.dumps(to_save))
    # Feedback
    print(f'Saved to {file_name}.json!')
