# -*- coding:utf-8 -*-
from Dataset import Dataset
import torch.utils.data as data
import torch
# from SNN import Model
# from SPP import Model
from DeepMIR import Model
import time
import datetime
import csv

timestamp = time.strftime("%Y%m%d-%H%M%S")
now = datetime.datetime.now()

# Define the path to the saved model file
# model_path = '../model/SNN.pth'  # SNN
# model_path = '../model/SPP.pth'  # SPP
model_path = '../model/DeepMIR.pth'

test_dataset = Dataset('../data/reftest_Binary.npy')
test_dataloader = data.DataLoader(test_dataset, shuffle=False, batch_size=12)
# Load the saved model state dictionary
state_dict = torch.load(model_path)
model = Model()
model.load_state_dict(state_dict)
model.eval()


# Specify the CSV file path
csv_file = '../res/DeepMIR_Binary.csv'


with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)

    for epoch in range(1):
        model.eval()

        for i, data in enumerate(test_dataloader, 0):
            inputs = data

            outputs = model(inputs)

            top_values, top_indices = torch.topk(outputs.view(-1), k=12)

            # Prepare the row data as a list
            row_data = []
            for j in range(12):
                row_data.extend([lib[top_indices[j]], float(top_values[j])])

            # Write the row data to the CSV file
            writer.writerow(row_data)