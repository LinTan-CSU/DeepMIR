import torch
# from ModelA import Model
# from ModelB import Model
from DeepMIR import Model
from Dataset import Dataset
import torch.utils.data as data
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import logging
import datetime
import numpy as np
import random

def predict(model_path, test_dataset, batch_size, lib)
    state_dict = torch.load(model_path)
    model = Model()
    model.load_state_dict(state_dict)
    model.eval()

    test_dataset = Dataset(test_dataset)
    test_dataloader = data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    csv_file = '../res/DeepMIR.csv'
    with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)

    for epoch in range(1):
        model.eval()
        results = []
        for i, data in enumerate(test_dataloader, 0):
            inputs = data
            outputs = model(inputs)

            top_values, top_indices = torch.topk(outputs.view(-1), k=batch_size)

            # Prepare the row data as a list
            row_data = []
            for j in range(batch_size):

                row_data.extend([lib[top_indices[j]], float(top_values[j])])

            # Write the row data to the CSV file
            writer.writerow(row_data)
  
    return row_data

if __name__ == '__main__':
    model_path = '../model/DeepMIR.pth'
    test_dataset = Dataset('../data/Quinary.npy')
    batch_size = 12
    lib = ['1,2-dichloroethane', '1-butanol', 'acetonitrile', 'cyclohexane', 'dichloromethane', 'diethylene_glycol_dimethyl_ether', 'ethanol', 'hexane', 'isopropyl_alcohol', 'methanol', 'toluene', 'trichloromethane']
    predict(model_path, test_dataset, batch_size, lib)
