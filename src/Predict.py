import pandas as pd
import torch
# from DeepRaman import Model
# from SCNN-DCNN import Model
from DeepMIR import Model
from Dataset import Dataset
import torch.utils.data
import csv


def predict(model_path, test_dataset_path, lib):
    """
    * Prediction
    *
    * Attributes
    * ----------
    * model_path : File path for storing the model
    * test_dataset_path : File path for storing the test dataset
    * lib : List of names of reference components
    *
    * Returns
    * -------
    * results : the predicted results
    """
    state_dict = torch.load(model_path)
    model = Model()
    model.load_state_dict(state_dict)
    model.eval()
    test_dataset = Dataset(test_dataset_path)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=len(lib))

    csv_file = '../res/DeepMIR.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)

        for i, data in enumerate(test_dataloader, 0):
            inputs = data
            outputs = model(inputs)

            top_values, top_indices = torch.topk(outputs.view(-1), k=len(lib))

            # Prepare the row data as a list
            row_data = []
            for j in range(len(lib)):
                row_data.extend([lib[top_indices[j]], float(top_values[j])])

            # Write the row data to the CSV file
            writer.writerow(row_data)
    results = pd.read_csv(csv_file, header=None)

    return results


if __name__ == '__main__':
    model_path = '../model/DeepMIR.pth'
    test_dataset_path = '../data/Quinary.npy'
    lib = ['1,2-dichloroethane', '1-butanol', 'acetonitrile', 'cyclohexane', 'dichloromethane',
           'diethylene_glycol_dimethyl_ether', 'ethanol', 'hexane', 'isopropyl_alcohol', 'methanol', 'toluene',
           'trichloromethane']
    results = predict(model_path, test_dataset_path, lib)
    print(results)
