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
#from plainModify import verification_return_variables
from plots import perform_identification_points
import bisect

# Doesn't work correctly, but not necessary.
def projected_finish(t_start, current_epoch, last_epoch):
    t_now = time.time()
    t_lapsed = t_now - t_start
    an_hour = 3600
    time_per_epoch = t_lapsed/current_epoch
    t_end = time_per_epoch * (last_epoch - current_epoch)

    ends_in_hrs = str(round(t_end/3600, 2))
    ends_at_utc = t_start + an_hour + t_end
    ends_at = datetime.utcfromtimestamp(ends_at_utc).strftime('%H:%M')

    print(f'Ends at {ends_at} ({ends_in_hrs} hrs)')


def print_current_time():
    t = time.time() + 3600
    current_t = datetime.utcfromtimestamp(t).strftime('%H:%M')
    print(f'Start at: {current_t}')

# Redundant class, just here for backward compatibility
def unbatch(X, Y):
    x = X[0]
    y = Y[0].copy()
    if len(X) > 1:  # X values are torch.tensor
        for batch in X[1:]:
            x = torch.cat([x, batch], dim=0)
    if len(Y) > 1:  # whlie Y values are lists of ints
        for batch in Y[1:]:
            y += batch
    return x, y


class Model(nn.Module):
    def __init__(self, input_size, layers):
        # Should it include a dropout layer?
        super().__init__()
        self.output_size = layers[-1]
        layerlist = []

        for layer in layers[:-1]:
            layerlist.append(nn.Linear(input_size, layer))
            layerlist.append(nn.ReLU(inplace=True))
            input_size = layer

        # Do last layer without activation
        layerlist.append(nn.Linear(input_size, layers[-1]))

        self.fc = nn.Sequential(*layerlist)

    def forward(self, x):
        x = self.fc(x)
        return x

    def get_output_size(self):
        return self.output_size


def transform_model(db: list, model_path: str):
    assert model_path in ['models/32_for_b1', 'models/32_for_b2', 'models/64_for_b1', 'models/64_for_b2', 'models/96_for_b1', 'models/96_for_b2', 'models/128_for_b1', 'models/128_for_b2', 'models/160_for_b1', 'models/160_for_b2']
    if model_path == 'models/32_for_b1' or model_path == "models/32_for_b2":
        layers_in = list(np.linspace(512, 32, 4, dtype=int))
    elif model_path == 'models/64_for_b1' or model_path == "models/64_for_b2":
        layers_in = list(np.linspace(512, 64, 4, dtype=int))
    elif model_path == 'models/96_for_b1' or model_path == "models/96_for_b2":
        layers_in = list(np.linspace(512, 96, 4, dtype=int))
    elif model_path == 'models/128_for_b1' or model_path == "models/128_for_b2":
        layers_in = list(np.linspace(512, 128, 4, dtype=int))
    elif model_path == 'models/160_for_b1' or model_path == "models/160_for_b2":
        layers_in = list(np.linspace(512, 160, 4, dtype=int))

    model = Model(512, layers_in[1:])
    model.load_state_dict(torch.load(model_path))
    if isinstance(db, list):  # db is a full DB, and not a single remove_single_sample_persons
        ndb = []
        with torch.no_grad():
            for person in db:
                plist = []
                for sample in person:
                    # convert a sample to tensor, take through model, back to
                    # numpy, append to plist.
                    plist.append(model(torch.FloatTensor(sample)).numpy())
                ndb.append(plist)
        return ndb
    else:  # received single sample
        return

# Take db through ANN model
def transform_with_model(db: list, model):
    if isinstance(db, list):  # db is a full DB, and not a single remove_single_sample_persons
        ndb = []
        with torch.no_grad():
            for person in db:
                plist = []
                for sample in person:
                    # convert a sample to tensor, take through model, back to
                    # numpy, append to plist.
                    plist.append(model(torch.FloatTensor(sample)).numpy())
                ndb.append(plist)
        return ndb
    else:  # received single sample
        return


# transform float DB to integer
def transform_to_int(db, n:int):
    #transform DB into n integers
    udb, _ = unroll_db(db)
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
            person.append(intsample)
        newdb.append(person)
    return newdb, translate

# Redundant class, just here for backward compatibility
def compute_EER(X, Y):
    # X is increasing, Y is decreasing
    for i in range(len(X)):
        if X[i] > Y[i]:
            break
    bestX = X[i-1] + (X[32]-X[31])/2
    bestY = Y[i-1] + (Y[32]-Y[31])/2
    return str(round((bestX + bestY)/2*100, 2))


# Redundant class, just here for backward compatibility
class SaveModel():
    def __init__(self, title: str):
        if os.path.abspath(os.getcwd()) == '/Users/jonasolafsson/Documents/speciale/scripts':
            results_path = '../biometrics.nosync/ANN_results/'
        else:
            results_path = '../ANN_results/'
        today = datetime.now()
        preamble = f'0{today.month}_{today.day}_{today.hour}h_{today.minute}m_{today.second}s_'
        self.path = results_path + preamble + title
        os.mkdir(self.path)
        self.path = self.path + '/'
        self.EER_verification = 'Not computed yet, run DET first'
        self.EER_identification = 'Not computed yet, run DET first'

    def get_path(self):
        return self.path

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


    def do_DET(self, db: list, resolution=100):
        gens, imps = verification_return_variables(db, 'float')
        points = resolution
        thresholds = list(np.linspace(min(imps), max(gens), points))
        # FMR: imposter < threshold in %
        FMR = [sum(imp < threshold for imp in imps)/len(imps)*100 for threshold in thresholds]
        # FNMR: gen > threshold in %
        FNMR = [sum(gen > threshold for gen in gens)/len(gens)*100 for threshold in thresholds]

        self.EER_verification = compute_EER(FMR, FNMR)

        sep = ','
        str = ''
        for row in range(points):
            str += f'{FMR[row]}{sep}{FNMR[row]}\n'
        file = open(self.path + 'DET-ver.csv', 'w+')
        file.write(str)
        file.close()

        plt.loglog(FMR, FNMR, '-b')
        plt.xlabel('False Math Rate (in %)')
        plt.ylabel('False Non-Math rate (in %)')
        plt.xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20])
        plt.yticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20])
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
        plt.grid()
        plt.savefig(self.path + 'DET-ver.pdf', bbox_inches='tight')
        plt.clf()

    def do_DET_ID(self, db: list, impostors: list, resolution=100):
        FPIR, FNIR = perform_identification_points(db, impostors, 'float')
        max_th = max([i for i in FNIR if i != 10])  # 10 is the PH dist if not rank1
        thresholds = list(np.linspace(min(FPIR), max_th, resolution))
        if thresholds[0] == 0:
            thresholds[0] = 1e-3
        # FMR: imposter < threshold in %
        X = [sum(imp < threshold for imp in FPIR)/len(FPIR)*100 for threshold in thresholds]
        # FNMR: gen > threshold in %
        Y = [sum(gen > threshold for gen in FNIR)/len(FNIR)*100 for threshold in thresholds]

        self.EER_identification = compute_EER(X,Y)

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

    def save_model(self, model):
        torch.save(model.state_dict(), self.path+'model')
