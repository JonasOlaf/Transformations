import pickle
import os
from os import path, listdir


def get_path():
    folder = '../biometrics.nosync/serialized_data/'
    return folder


def save_db(db: list, filename: str):
    folder = get_path()
    file = folder + filename + ".pkl"
    if path.exists(file):
        print(f'{filename} already exists. Do you want to overwrite? [y/n]')
        answer = input()
        if answer != 'y':
            print('Action aborted')
            return
    open_file = open(file, "wb")
    pickle.dump(db, open_file)
    open_file.close()


def get_db(filename: str):
    folder = get_path()
    file = folder + filename + ".pkl"
    if not path.exists(file):
        print(f'File \"{filename}\" does not exist. Serialized files are:')
        for name in listdir('../biometrics.nosync/serialized_data'):
            print(name[:-4])
        return
    open_file = open(file, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()
    return loaded_list


def list_dbs():
    folder = get_path()
    for name in listdir(folder):
        if not name.startswith('.'):  # ignores hidden files
            print(name[:-4])
    return
