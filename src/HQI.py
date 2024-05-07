import numpy as np
from Dataset import Dataset
import torch.utils.data as data
import torch
import csv

def HQI(test_dataset, batch_size, lib):
    test_dataset = Dataset(test_dataset)
    test_dataloader = data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    csv_file = '../res/HQI.csv'
    outputs = []

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)

        for epoch in range(1):
            for i, data in enumerate(test_dataloader, 0):
                inputs = data
                test = data[0, 1, :]
                row_data = []
                HQIs = []
                for j in range(batch_size):
                    reference = data[j, 0, :]
                    HQI = np.dot(reference, test) ** 2 / (np.dot(reference, reference) * np.dot(test, test))
                    HQIs.append(HQI)
                  
                outputs = torch.tensor(HQIs)
                top_values, top_indices = torch.topk(outputs.view(-1), k=batch_size)
          
                for k in range(batch_size):
                    row_data.extend([lib[top_indices[k]], float(top_values[k])])
                writer.writerow(row_data)
    return row_data

if __name__ == '__main__':
    test_dataset = '../data/Quinary.npy'
    batch_size = 12
    lib = ['1,2-dichloroethane', '1-butanol', 'acetonitrile', 'cyclohexane', 'dichloromethane', 'diethylene_glycol_dimethyl_ether', 'ethanol', 'hexane', 'isopropyl_alcohol', 'methanol', 'toluene', 'trichloromethane']
    RM(test_dataset, batch_size, lib)    
