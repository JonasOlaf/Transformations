import sys
sys.path.append('../functions/')
from DataSerilization import get_db
from Transforms import conv_to_torch, conv_from_torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from ANN_helpers import projected_finish, print_current_time, unbatch, transform_with_model
from ANN_savemodel import SaveModel2
from TripletLoss import AnchorPicker

import pdb


# SETUP

# Name of folder to save results in
ratio_train = 0.8
folder_name = f'to_32_b1'
Description = 'Reducing to 32, training on first half, testing on second.'

Batch_size = 300
epochs = 8
learning_rate = 0.001
margin_triplet = 0.5
layers_in = list(np.linspace(512, 32, 4, dtype=int))[1:]
#layers_in = list(np.logspace(np.log10(512), np.log10(128)+1, 4, dtype=int))[1:]
db_name = 'vgg_train_8, 4 batches'


db = get_db('vgg/4/b1')
db += get_db('vgg/4/b2')
db3 = get_db('vgg/4/b3')
db3 += get_db('vgg/4/b4')
ldb3 = len(db3)
db_gens = db3[:int(ldb3*0.8)]
db_imps = db3[-int(ldb3*0.2):]


db_gens = [person[:3] for person in db_gens]
db_imps = [person[:3] for person in db_imps]


X_train, Y_train, _, _ = conv_to_torch(db, rate=1, batch_size=Batch_size)

# ---------
no_batches = len(X_train)
len_train_batch = len(X_train[0])
persons_train_batch = len(list(set(Y_train[0])))
anchors = AnchorPicker(X_train, Y_train, layers_in[-1])


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


# Initialise network
torch.manual_seed(42)
model = Model(X_train[0].shape[1], layers_in)
criterion = nn.TripletMarginLoss(margin=margin_triplet)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



# TRAINING #####
losses = []
t1 = time.time()
print_current_time()
output_size = model.get_output_size()
epochs_actual = epochs  # to note in case we exit the training early
sm = SaveModel2(folder_name)

print('Running DET')
time_preDET = time.time()
small_dbgens = transform_with_model(db_gens, model)
small_dbimps = transform_with_model(db_imps, model)
sm.do_running_DET_ID(small_dbgens, small_dbimps, epoch='pre')
print(f'DET took {round((time.time()-time_preDET)/60,2)} minutes.')

for epoch in range(epochs):
    # train batches
    for batch in range(no_batches):
        print(f'Epoch {epoch+1}/{epochs} \t Train-Batch\t{batch+1}/{no_batches}')
        pred_anchors, pred_positives, pred_negatives = anchors.get_anchors(batch=batch, model=model)
        # compute loss and optimize model for batch
        loss = criterion(pred_anchors, pred_positives, pred_negatives)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses.append(loss.item())  # only append last loss per epoch


    #db_train_sq = conv_from_torch(model_x_train, y_train)
    #db_test_sq = conv_from_torch(model_x_test, y_test)

    trans_dbgens = transform_with_model(db_gens, model)
    trans_dbimps = transform_with_model(db_imps, model)
    sm.do_running_DET_ID(trans_dbgens, trans_dbimps, epoch=epoch)
    sm.save_model(model, epoch)

dt = time.time()-t1



def format_variables_to_string():
    string = folder_name + '\n'
    string += Description + '\n\n'

    string += f'db:\t\t\t{db_name}\n'
    string += f'Epochs:\t\t\t{epochs}\n'
    string += f'Epochs(run): \t\t{epochs_actual}\n'
    string += f'Batch size:\t\t{Batch_size}\n'
    string += f'Layers:\t\t\t{layers_in}\n'
    string += f"Learning Rate:\t\t{learning_rate}\n"
    string += f'Margin:\t\t\t{margin_triplet}\n'
    string += f'Time (mins):\t\t{round(dt/60,2)}\n'
    #identification_rate_test = identification_return(trans_dbimps[:1000], 'float')
    #string += f'Identification (db_gens[:1k]):\t{round(identification_rate_test,2)}%\n'
    #string += f'Ratio:\t\t\t{ratio_train*100}%\n\n\n'
    string += f'No. train batches:\t\t{no_batches}\n'
    string += f'Persons in train batches:\t{persons_train_batch}\n'
    string += f'Samples in train batches:\t{len_train_batch}\n\n'

    string += f'\nPersons in gens DB:\t\t{len(db_gens)}\n'
    string += f'Persons in imps DB:\t\t{len(db_imps)}\n'
    string += f'EER (identification) in %\t{sm.EER_identification}'

    return string



sm.save_info(format_variables_to_string())
#sm.save_losses([losses, losses_test])
sm.save_db(transform_with_model(db, model), 'traindata')
sm.save_db(trans_dbgens, 'db_gens')
sm.save_db(trans_dbimps, 'db_imps')
np.savetxt(sm.path + 'losses.csv', losses, delimiter=',')
#sm.save_model(model)
#sm.save_csv([running_identification_train, running_identification_test], 'identifications')


# Compute DET values
