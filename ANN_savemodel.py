import pickle
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
#
#from plainModify import verification_return_variables


class SaveModel2():
    def __init__(self, title: str):
        if os.path.abspath(os.getcwd()) == '/Users/jonasolafsson/Documents/speciale/scripts':
            results_path = '../biometrics.nosync/ANN_results/'
        else:
            results_path = ''# '../ANN_results/'
        today = datetime.now()
        preamble = f'0{today.month}_{today.day}_{today.hour}h_{today.minute}m_{today.second}s_'
        self.path = results_path + preamble + title
        os.mkdir(self.path)
        os.mkdir(self.path+'/DETs')
        os.mkdir(self.path+'/models')
        self.path = self.path + '/'
        self.DETpath = self.path + 'DETs/'
        self.modelpath = self.path + 'models/'
        # self.EER_verification = 'Not computed yet, run DET first'
        self.EER_identification = 'Not computed yet, run DET first'

    def get_path(self):
        return self.path

    def compute_EER(self, X, Y):
        for i in range(len(X)):
            if X[i] > Y[i]:
                return 0.5*(X[i]+Y[i])

    def save_info(self, string: str):
        # save input string to file 'info.txt'
        txt = open(self.path + 'info.txt', 'w+')
        txt.write(string)
        txt.close()

    def save_losses(self, losses: list, sep=','):
        test_included = isinstance(losses[0], list)
        str = ''
        length = len(losses[0])

        for row in range(length):
            if test_included:
                str += f'{row}{sep}{losses[0][row]}{sep}{losses[1][row]}\n'
            else:
                str += f'{row}{sep}{loss[0][row]}\n'
        file = open(self.path + 'losses.csv', 'w+')
        file.write(str)
        file.close()
        self.plot_losses('losses.csv')

    def save_csv(self, lists: list, name: str, sep=','):
        str = ''

        for row in range(len(lists[0])):
            str += f'{row}{sep}{lists[0][row]}{sep}{lists[1][row]}\n'

        file = open(self.path + name+'.csv', 'w')
        file.write(str)
        file.close()


    # def do_DET(self, db: list, resolution=200):
    #     gens, imps = verification_return_variables(db, 'float')
    #     points = resolution
    #     thresholds = list(np.linspace(min(imps), max(gens), points))
    #     # FMR: imposter < threshold in %
    #     FMR = [sum(imp < threshold for imp in imps)/len(imps)*100 for threshold in thresholds]
    #     # FNMR: gen > threshold in %
    #     FNMR = [sum(gen > threshold for gen in gens)/len(gens)*100 for threshold in thresholds]
    #
    #     self.EER_verification = self.compute_EER(FMR, FNMR)
    #
    #     sep = ','
    #     str = ''
    #     for row in range(points):
    #         str += f'{FMR[row]}{sep}{FNMR[row]}\n'
    #     file = open(self.path + 'DET-ver.csv', 'w+')
    #     file.write(str)
    #     file.close()
    #
    #     plt.loglog(FMR, FNMR, '-b')
    #     plt.xlabel('False Math Rate (in %)')
    #     plt.ylabel('False Non-Math rate (in %)')
    #     plt.xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20])
    #     plt.yticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20])
    #     plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    #     plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    #     plt.grid()
    #     plt.savefig(self.path + 'DET-ver.pdf', bbox_inches='tight')
    #     plt.clf()

    def do_DET_ID(self, db: list, impostors: list, resolution=200):
        FPIR, FNIR = perform_identification_points(db, impostors, 'float')
        max_th = max([i for i in FNIR if i != 10])  # 10 is the PH dist if not rank1
        thresholds = list(np.linspace(min(FPIR), max_th, resolution))
        if thresholds[0] == 0:
            thresholds[0] = 1e-3
        # FMR: imposter < threshold in %
        X = [sum(imp < threshold for imp in FPIR)/len(FPIR)*100 for threshold in thresholds]
        # FNMR: gen > threshold in %
        Y = [sum(gen > threshold for gen in FNIR)/len(FNIR)*100 for threshold in thresholds]

        #self.EER_identification = compute_EER(X,Y)

        sep = ','
        str = ''
        for row in range(resolution):
            str += f'{X[row]}{sep}{Y[row]}\n'
        file = open(self.path + 'DET-ID.csv', 'w+')
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
        plt.savefig(self.path + 'DET-ID.pdf', bbox_inches='tight')
        plt.clf()

    def euclidean_distance(self, a: list, b: list):
        a = np.array(a)
        b = np.array(b)
        return np.sum(np.square(a-b))

    def perform_identification_points(self, db: list, impostors: list, feature_type: str, rank=1):
        # FPIR / Type 1:
        # Fraction where an imposter is below threshold of nearest sample
        #
        # FNIR / Type 2:
        # Genuine returns an incorrect sample OR distance is above threshold
        FPIR = []
        FNIR = []

        # FPIR distances
        count = 0
        for person in impostors:
            for probe in person:  # Test with every sample of every person
                min_dist = 100
                for dbperson in db:
                    dist = self.euclidean_distance(dbperson[0], probe)
                    if dist < min_dist:
                        min_dist = dist
                FPIR.append(min_dist)
            count += 1
            if count % 500 == 0:
                print(f'DET Imposter at {count}/{len(impostors)}')
        # FNIR distances, genuine distances
        count = 0
        for subject in range(len(db)):
            for sample in range(1, len(db[subject])):  # take every sample of that person
                gen_dist = self.euclidean_distance(db[subject][sample], db[subject][0])
                is_rank_k = True
                subjects_closer_than_gen = 0
                for ref in range(len(db)):
                    if subject == ref:  # person, not sample
                        continue
                    dist = self.euclidean_distance(db[subject][sample], db[ref][0])
                    if dist < gen_dist:
                        subjects_closer_than_gen += 1
                        if subjects_closer_than_gen >= rank:
                            is_rank_k = False
                            break
                if is_rank_k:
                    FNIR.append(gen_dist)
                else:
                    FNIR.append(10)  # high value, above threshold to mark as failure
            count += 1
            if count % 500 == 0:
                print(f'DET Genuines at {count}/{len(db)}')
        return FPIR, FNIR

    def do_running_DET_ID(self, db: list, impostors: list, epoch: int, resolution=200):
        FPIR, FNIR = self.perform_identification_points(db, impostors, 'float')
        max_th = max([i for i in FNIR if i != 10])  # 10 is the PH dist if not rank1
        thresholds = list(np.linspace(min(FPIR), max_th, resolution))
        if thresholds[0] == 0:
            thresholds[0] = 1e-3
        # FMR: imposter < threshold in %
        X = [sum(imp < threshold for imp in FPIR)/len(FPIR)*100 for threshold in thresholds]
        # FNMR: gen > threshold in %
        Y = [sum(gen > threshold for gen in FNIR)/len(FNIR)*100 for threshold in thresholds]

        self.EER_identification = self.compute_EER(X,Y)

        sep = ','
        str = ''
        for row in range(resolution):
            str += f'{X[row]}{sep}{Y[row]}\n'
        file = open(f'{self.DETpath}DET-ID({epoch}).csv', 'w+')
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
        plt.savefig(f'{self.DETpath}{epoch}-DET-ID.pdf', bbox_inches='tight')
        plt.clf()


    def plot_losses(self, losses:str):
        loss = np.genfromtxt(self.path + losses, delimiter=',')
        plt.plot(loss[:,0], loss[:,1], '-b', label='train')
        plt.plot(loss[:,0], loss[:,2], '-r', label='test')
        plt.legend(loc='upper right')
        plt.xlabel('Generation')
        plt.ylabel('Loss')
        plt.savefig(self.path + 'loss_plot.pdf', bbox_inches='tight')
        plt.clf()

    def save_db(self, db: list, filename: str):
        filepath = self.path + filename +'.pkl'
        file = open(filepath, 'wb')
        pickle.dump(db, file)
        file.close()

    def save_model(self, model, epoch):
        torch.save(model.state_dict(), self.modelpath+f'{epoch}-model')
