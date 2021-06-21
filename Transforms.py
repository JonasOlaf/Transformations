import torch
from sklearn.utils import shuffle
from Helpers import unroll_db
import bisect
import numpy as np



def remove_single_sample_persons(db):
    # remove people with only one sample
    ndb = db.copy()
    for i in range(len(ndb)-1, -1, -1):  # go oppositely through the list
        if len(ndb[i]) < 2:
            del ndb[i]
    return ndb



def conv_to_torch(db, rate=0.8, batch_size=0):
    # convert db to tensor of
    # rate: 80% of persons (not samples) are in training data. Since data is
    # random, shuffle is performed after split.
    # no_batches: number of batches. Splits people into (approx) equally large
    # batches.
    ndb = remove_single_sample_persons(db)  # removes people with one sample
    ndb = shuffle(ndb, random_state=42)
    persons = len(ndb)
    len_train = int(persons*rate)
    len_test = int(persons*(1-rate))

    if batch_size == 0:
        batch_size = len_train-1  # if not specified, use one batch

    ignore_testdata = rate == 1

    no_batches = len_train // batch_size
    if no_batches == 0:
        no_batches = 1

    if batch_size > len_test:
        batch_size = len_test - 1
    no_test_batches = len_test // batch_size

    X_train = [[] for i in range(no_batches)]
    X_test = [[] for i in range(no_test_batches)]
    Y_train = [[] for i in range(no_batches)]
    Y_test = [[] for i in range(no_test_batches)]

    for i, person in enumerate(ndb):
        batch = i%no_batches
        if not ignore_testdata:
            test_batch = i%no_test_batches

        for sample in person:
            if i <= len_train:
                X_train[batch].append(sample)
                Y_train[batch].append(i)
            else:
                X_test[test_batch].append(sample)
                Y_test[test_batch].append(i)

    # shuffle batched training data
    for i, _ in enumerate(X_train):
        X_train[i], Y_train[i] = shuffle(X_train[i], Y_train[i], random_state=42)
        X_train[i] = torch.FloatTensor(X_train[i])
    for i, _ in enumerate(X_test):
        X_test[i], Y_test[i] = shuffle(X_test[i], Y_test[i], random_state=42)
        X_test[i] = torch.FloatTensor(X_test[i])

    return X_train, Y_train, X_test, Y_test


def conv_from_torch(x, y):
    # converts
    no_persons = len(list(set(y)))  # number of persons in db
    place = {}  # dictionary to map x to persons
    db = []
    for i, sample in enumerate(x):
        if y[i] not in place:  # if new person
            place[y[i]] = len(db)
            db.append([])
        db[place[y[i]]].append(sample.numpy())  # converst from torch to numpy
    return db


def compute_int_bins(db, n:int):
    #transform DB into n integers
    #udb, _ = unroll_db(db)

    #To only use enrolled subjects for conversion
    udb = []
    for subject in db:
        udb.append(subject[0])

    translate = []
    for i in range(len(udb[0])):
        translate.append([])
        L = [sample[i] for sample in udb]
        L.sort()
        for j in range(1,n):
            translate[i].append(L[int(len(udb)/n*j)])

    newdb = []
    for subject in db: # should be len(translate)
        person = []
        for sample in subject:
            intsample = []
            for i, value in enumerate(sample):
                intsample.append(bisect.bisect_left(translate[i], value))
            person.append(np.array(intsample))
        newdb.append(person)
    return translate

def transform_to_int(db, translate):
        newdb = []
        for subject in db: # should be len(translate)
            person = []
            for sample in subject:
                intsample = []
                for i, value in enumerate(sample):
                    intsample.append(bisect.bisect_left(translate[i], value))
                person.append(np.array(intsample))
            newdb.append(person)
        return newdb



def transform_to_bin(db, n:int):
    assert n in [3,4]

    translate = compute_int_bins(db, n)
    db = transform_to_int(db, translate)
    newdb = []
    for subject in db:
        person = []
        for sample in subject:
            converted_sample = ''
            if n == 3:
                for value in sample:
                    if value == 0:
                        converted_sample += '00'
                    if value == 1:
                        converted_sample += '01'
                    if value == 2:
                        converted_sample += '11'
            if n == 4:
                for value in sample:
                    if value == 0:
                        converted_sample += '000'
                    if value == 1:
                        converted_sample += '001'
                    if value == 2:
                        converted_sample += '011'
                    if value == 3:
                        converted_sample += '111'
            person.append(converted_sample)
        newdb.append(person)
    return newdb


def transform_to_int_single(sample, translate):
        intsample = []
        for i, value in enumerate(sample):
            intsample.append(int(bisect.bisect_left(translate[i], value)))
        return intsample


def transform_to_bin_single(float_sample, n:int, translate):
    assert n in [3,4]
    sample = transform_to_int_single(float_sample, translate)
    converted_sample = ''
    if n == 3:
        for value in sample:
            if value == 0:
                converted_sample += '00'
            if value == 1:
                converted_sample += '01'
            if value == 2:
                converted_sample += '11'
    if n == 4:
        for value in sample:
            if value == 0:
                converted_sample += '000'
            if value == 1:
                converted_sample += '001'
            if value == 2:
                converted_sample += '011'
            if value == 3:
                converted_sample += '111'
    return converted_sample


#
