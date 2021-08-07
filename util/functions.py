import os

def get_absolute_path(path):
    return os.path.join(os.getcwd(), '../', path)