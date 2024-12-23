import os, json


def load(path):
    # Load json file
    try:
        to_load = json.load(open(path, 'r'))
        return to_load
    except FileNotFoundError:
        print("LOAD_ERROR: File doesn't exist!\n")

def save(path, to_save={}):
    # Save into json file
    open(path, 'w').write(json.dumps(to_save))    
