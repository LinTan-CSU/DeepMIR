# -*- coding:utf-8 -*-
from Dataset import Dataset
import torch.utils.data as data
import torch
from DeepMIR import Model
# from ModelA import Model
# from ModelB import Model
import csv
import numpy as np

model_path = '../model/DeepMIR.pth'
# model_path = '../model/ModelA.pth'
# model_path = '../model/ModelB.pth'
test_dataset = Dataset('../data/reftest_pigment_52_airPLS1000_WS100_max_min.npy')
test_dataloader = data.DataLoader(test_dataset, shuffle=False, batch_size=11)
state_dict = torch.load(model_path)
model = Model()
model.load_state_dict(state_dict)
model.eval()
lib = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

csv_file = '../res/DeepMIR_pigment_52_airPLS1000_WS100_max_min.csv'
# csv_file = '../res/ModelA_pigment_52_airPLS1000_WS100_max_min.csv'
# csv_file = '../res/ModelB_pigment_52_airPLS1000_WS100_max_min.csv'

with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)

    for epoch in range(1):
        model.eval()

        for i, data in enumerate(test_dataloader, 0):
            inputs = data

            outputs = model(inputs)

            top_values, top_indices = torch.topk(outputs.view(-1), k=11)

            # Prepare the row data as a list
            row_data = []
            for j in range(11):
                row_data.extend([lib[top_indices[j]], float(top_values[j])])

            # Write the row data to the CSV file
            writer.writerow(row_data)
