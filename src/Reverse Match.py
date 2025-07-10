import numpy as np
from scipy.signal import find_peaks
from Dataset import Dataset
import torch.utils.data
import torch
import csv
import pandas as pd

def RM(test_dataset_path, test_labels):
    """
    * RM
    *
    * Attributes
    * ----------
    * test_dataset_path : File path for storing the test dataset
    * test_labels : List of names of reference components
    *
    * Returns
    * -------
    * results : the predicted results
    """
    test_dataset = Dataset(test_dataset_path)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=len(test_labels))
    csv_file = '../res/RM.csv'

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)

        for epoch in range(1):

            for i, data in enumerate(test_dataloader, 0):
                test = data[0, 1, :]
                row_data = []
                CCs = []
                for j in range(len(test_labels)):
                    reference = data[j, 0, :]
                    peaks1, _ = find_peaks(reference, height=0.05)
                    correlation_coefficient = np.corrcoef(reference[peaks1], test[peaks1])[0, 1]
                    CCs.append(correlation_coefficient)
                outputs = torch.tensor(CCs)
                top_values, top_indices = torch.topk(outputs.view(-1), k=len(test_labels))

                for k in range(len(test_labels)):
                    row_data.extend([test_labels[top_indices[k]], float(top_values[k])])
                writer.writerow(row_data)
    results = pd.read_csv(csv_file, header=None)

    return results


if __name__ == '__main__':
    test_dataset = '../data/Quinary.npy'
    test_labels = ['1,2-dichloroethane', '1-butanol', 'acetonitrile', 'cyclohexane', 'dichloromethane',
           'diethylene_glycol_dimethyl_ether', 'ethanol', 'hexane', 'isopropyl_alcohol', 'methanol', 'toluene',
           'trichloromethane']
    results = RM(test_dataset, test_labels)
    print(results)
