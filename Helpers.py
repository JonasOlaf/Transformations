import numpy as np
import time


def unroll_db(db: list):
    # turns 3d db into 2d, by ignoring different people
    data = []
    # Number of samples per person, to ununroll db.
    indices = []
    for subject in db:
        counter = 0
        for sample in subject:
            data.append(sample)
            counter += 1
        indices.append(counter)
    return data, indices


def inroll_db(db: list, indices: list):
    inroll_db = []
    next_index = 0
    # number of people
    for no_samples in indices:
        # number of samples
        person = []
        for i in range(no_samples):
            person.append(db[next_index])
            next_index += 1
        inroll_db.append(person)
    return inroll_db


def remove_rows_db(db: list, rows: list):
    database = []
    for subject in db:
        person = []
        for sample in subject:
            new_sample = np.delete(sample, rows)
            person.append(new_sample)
        database.append(person)
    return database


def transpose_data(data: list):
    return list(map(list, zip(*data)))


def twod_list_to_csv(data: list, filename: str):
    import pandas as pd
    path = "../csv_results/"
    df = pd.DataFrame(data)
    path_and_name = path + filename + ".csv"
    df.to_csv(path_and_name, header=False)




# Identification scrips
def compute_distance(a, b, typ:str):
    d = -1
    if typ == 'bin':
        d = int_HD(a, b)
    elif typ == 'float':
        d = euclidean_distance(a, b)
    elif typ == 'int':
        d = euclidean_distance(a, b)
    return d

def euclidean_distance(a: list, b: list):
    a = np.array(a)
    b = np.array(b)
    return np.sum(np.square(a-b))



def search_db_for_identification(db:list, probe, feature_type):
    distances = []
    for subject in db:
        # Test probe on first sample on every person
        distances.append(compute_distance(subject[0], probe, feature_type))
    return distances.index(np.min(distances)), np.min(distances)


def identification(db: list, feature_name: str, feature_type: str):
    # db format: db[arb_person][sample][dimension]
    # db = read_templates(feature_name, feature_type)
    t1 = time.time()
    rank1 = 0
    non = 0
    # enumerate every arb. person in db
    for i, subject in enumerate(db):
        # if a subject has more than one sample (SHOULD be everyone)
        if len(subject) > 1:
            for j in range(1, len(subject)):
                # test all samples except the 0th on all 0th entries in db
                d, dist = search_db_for_identification(db, subject[j], feature_type)
                if d == i:
                    rank1 += 1
                else:
                    #print(f'Failed identification. Identified: {d}, but was {i}')
                    print(f'id: {i},{j}\t identified as: {d},0\t distance: {dist}')
                    non += 1
    t2 = time.time()
    print(f'{feature_name} {feature_type} rank1: {round(rank1 / (rank1+non)*100,2)}%')
    run_time = str(round(t2-t1,2))
    print("Execution time: " + run_time + "s.")


def identification_return(db: list, feature_type: str):
    # db format: db[arb_person][sample][dimension]
    # db = read_templates(feature_name, feature_type)
    #t1 = time.time()
    rank1 = 0
    non = 0
    # enumerate every arb. person in db
    for i, subject in enumerate(db):
        # if a subject has more than one sample (SHOULD be everyone)
        if len(subject) > 1:
            for j in range(1, len(subject)):
                # test all samples except the 0th on all 0th entries in db
                d = search_db_for_identification(db, subject[j], feature_type)
                if d == i:
                    rank1 += 1
                else:
                    non += 1
    #t2 = time.time()
    #print(feature_name, feature_type, 'rank1: {}%'.format(rank1 / (rank1+non)*100))
    #run_time = str(round(t2-t1,2))
    #print("Execution time: " + run_time + "s.")
    return rank1 / (rank1+non)*100


def search_db_for_top_identification(db:list, probe, feature_type, k: int):
    distances = []
    for subject in db:
        # Test probe on first sample on every person
        distances.append(compute_distance(subject[0], probe, feature_type))
    # Rank-k indices
    ind = sorted(range(len(distances)), key = lambda sub: distances[sub])[:k]
    return ind


def rank_k_identification(db: list, feature_name: str, feature_type: str, k: int):
    # db format: db[arb_person][sample][dimension]
    # db = read_templates(feature_name, feature_type)
    t1 = time.time()
    rank1 = 0
    non = 0
    # enumerate every arb. person in db
    for i, subject in enumerate(db):
        # if a subject has more than one sample (SHOULD be everyone)
        if len(subject) > 1:
            for j in range(1, len(subject)):
                # test all samples except the 0th on all 0th entries in db
                indices = search_db_for_top_identification(db, subject[j], feature_type, k)
                if i in indices:#d == i:
                    rank1 += 1
                else:
                    non += 1
    t2 = time.time()
    print(feature_name, feature_type, 'rank1: {}%'.format(rank1 / (rank1+non)*100))
    run_time = str(round(t2-t1,2))
    print("Execution time: " + run_time + "s.")
