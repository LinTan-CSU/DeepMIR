import numpy as np
from scipy.signal import find_peaks
from Dataset import Dataset
import torch.utils.data as data
import torch
import csv

test_dataset = Dataset('../data/reftest_pigment_52_airPLS1000_WS100_max_min.npy')
test_dataloader = data.DataLoader(test_dataset, shuffle=False, batch_size=11)
lib = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
csv_file = '../res/RM_pigment_52_airPLS1000_WS100_max_min.csv'
outputs = []

with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)

    for epoch in range(1):

        for i, data in enumerate(test_dataloader, 0):
            inputs = data
            test = data[0, 1, :]
            row_data = []
            CCs = []
            for j in range(11):
                reference = data[j, 0, :]
                peaks1, _ = find_peaks(reference, height=0.05)
                correlation_coefficient = np.corrcoef(reference[peaks1], test[peaks1])[0, 1]
                CCs.append(correlation_coefficient)
            outputs = torch.tensor(CCs)
            top_values, top_indices = torch.topk(outputs.view(-1), k=11)
          
            for k in range(11):
                row_data.extend([lib[top_indices[k]], float(top_values[k])])
            writer.writerow(row_data)
