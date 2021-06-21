import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import pdb
import bisect

def perform_identification_points(db: list, impostors: list, feature_type: str, rank=1, dist_type='euclidean'):
    # FPIR / Type 1:
    # Fraction where an imposter is below threshold of nearest sample
    #
    # FNIR / Type 2:
    # Genuine returns an incorrect sample OR distance is above threshold
    assert dist_type in ['euclidean', 'manhattan']
    FPIR = []
    FNIR = []
    # FPIR distances
    count = 0
    print(f'DET Imposter at {0}/{len(impostors)}')
    for person in impostors:
        for probe in person:  # Test with every sample of every person
            min_dist = 100000
            for dbperson in db:
                if dist_type == 'euclidean':
                    dist = euclidean_distance(dbperson[0], probe)
                elif dist_type == 'manhattan':
                    dist = manhattan_distance(dbperson[0], probe)
                if dist < min_dist:
                    min_dist = dist
            FPIR.append(min_dist)
        count += 1
        if count % 500 == 0:
            print(f'DET Imposter at {count}/{len(impostors)}')
    print(f'DET Imposter at {len(impostors)}/{len(impostors)}')
    # FNIR distances, genuine distances
    count = 0
    for subject in range(len(db)):
        for sample in range(1, len(db[subject])):  # take every sample of that person
            #pdb.set_trace()
            if dist_type == 'euclidean':
                gen_dist = euclidean_distance(db[subject][sample], db[subject][0])
            elif dist_type == 'manhattan':
                gen_dist = manhattan_distance(db[subject][sample], db[subject][0])
            is_rank_k = True
            subjects_closer_than_gen = 0
            for ref in range(len(db)):
                if subject == ref:  # person, not sample
                    continue
                if dist_type == 'euclidean':
                    dist = euclidean_distance(db[subject][sample], db[ref][0])
                elif dist_type == 'manhattan':
                    dist = manhattan_distance(db[subject][sample], db[ref][0])
                if dist < gen_dist:
                    subjects_closer_than_gen += 1
                    if subjects_closer_than_gen >= rank:
                        is_rank_k = False
                        break
            if is_rank_k:
                FNIR.append(gen_dist)
            else:
                FNIR.append(10000)  # high value, above threshold to mark as failure
        count += 1
        if count % 500 == 0:
            print(f'DET Genuines at {count}/{len(db)}')
    print(f'DET Genuines at {len(db)}/{len(db)}')
    return FPIR, FNIR


def dist_curve(db, name='dists', ratio=0.8, resolution=50, delimiter=','):
    ldb = len(db)
    db_gens = db[:int(ldb*0.8)]
    db_imps = db[-int(ldb*0.2):]
    FPIR, FNIR = perform_identification_points(db_gens, db_imps, 'float')
    #FPIR = get_db('fpir_vgg1+2')
    #FNIR = get_db('fnir_vgg1+2')
    cats = np.linspace(0, max(FPIR)*1.1, resolution)

    FNIR.sort()
    FPIR.sort()
    fnir = []
    fpir = []
    for cat in cats:
        fnir.append(bisect.bisect_right(FNIR, cat))
        fpir.append(bisect.bisect_right(FPIR, cat))
    for i in range(len(fnir)):
        if i == 0:
            y = [fnir[0]]
            y1 = [fpir[0]]
        else:
            y.append(fnir[i]-fnir[i-1])
            y1.append(fpir[i]-fpir[i-1])
    y1 = [i *len(FNIR)/len(FPIR) for i in y1]
    s = ''
    for i in range(len(cats)):
        s += f'{cats[i]}{delimiter}{y[i]}{delimiter}{y1[i]}\n'
    file = open(f'dists/{name}.csv', 'w+')
    file.write(s)
    file.close()


def euclidean_distance(a: list, b: list):
    a = np.array(a)
    b = np.array(b)
    return np.sum(np.square(a-b))

def manhattan_distance(a: list, b: list):
    a = np.array(a)
    b = np.array(b)
    return np.sum(np.absolute(a-b))


def DET_identification(db: list, output_name: str, resolution=100, rank=1, dist_type='euclidean'):
    assert dist_type in ['euclidean', 'manhattan']
    db_gens = db[:int(0.8*len(db))]
    db_imps = db[int(0.8*len(db)):]
    FPIR, FNIR = perform_identification_points(db_gens, db_imps, 'float', rank, dist_type)
    max_th = max([i for i in FNIR if i != 10000])  # 10000 is the PH dist if not rank1
    thresholds = list(np.linspace(min(FPIR), max_th, resolution))
    if thresholds[0] == 0:
        thresholds[0] = 1e-3
    # FMR: imposter < threshold in %
    X = [sum(imp < threshold for imp in FPIR)/len(FPIR)*100 for threshold in thresholds]
    # FNMR: gen > threshold in %
    Y = [sum(gen > threshold for gen in FNIR)/len(FNIR)*100 for threshold in thresholds]

    sep = ','
    str = ''
    for row in range(resolution):
        str += f'{X[row]}{sep}{Y[row]}\n'
    file = open(f'det/{output_name}.csv', 'w+')
    file.write(str)
    file.close()

    plt.loglog(X, Y, '-b')
    plt.xlabel('False Positive Identification Rate (FPIR) (in %)')
    plt.ylabel('False Negative Identification Rate (FNIR) (in %)')
    plt.xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20])
    plt.yticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20])
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.grid()
    plt.savefig(f'det/{output_name}.pdf', bbox_inches='tight')
    plt.clf()
