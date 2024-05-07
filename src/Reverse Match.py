import numpy as np
from scipy.signal import find_peaks
from Dataset import Dataset
import torch.utils.data as data
import torch
import csv

def RM(test_dataset, batch_size, lib):
    test_dataset = Dataset(test_dataset)
    test_dataloader = data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    csv_file = '../res/RM.csv'
    outputs = []

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)

        for epoch in range(1):

            for i, data in enumerate(test_dataloader, 0):
                inputs = data
                test = data[0, 1, :]
                row_data = []
                CCs = []
                for j in range(batch_size):
                    reference = data[j, 0, :]
                    peaks1, _ = find_peaks(reference, height=0.05)
                    correlation_coefficient = np.corrcoef(reference[peaks1], test[peaks1])[0, 1]
                    CCs.append(correlation_coefficient)
                outputs = torch.tensor(CCs)
                top_values, top_indices = torch.topk(outputs.view(-1), k=batch_size)
          
                for k in range(batch_size):
                    row_data.extend([lib[top_indices[k]], float(top_values[k])])
                writer.writerow(row_data)
    return row_data

if __name__ == '__main__':
    test_dataset = '../data/test_dataset.npy'
    batch_size = 10
    lib = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    RM(test_dataset, batch_size, lib)
    
