import csv

def Metrics(output_file_path, labels_file_path, threshold=0.5):
    """
    * Metrics
    *
    * Attributes
    * ----------
    * output_file_path : File path for storing the output file
    * labels_file_path : File path for storing the labels file
    * threshold : threshold for predicted probabilities
    *
    * Returns
    * -------
    * TPR : the true positive rate of predicted results
    * TNR : the true negative rate of predicted results
    * ACC : the accuracy of predicted results
    """
    TP = []
    TN = []
    FP = []
    FN = []

    # Read the output file
    with open(output_file_path, 'r') as file:
        reader = csv.reader(file)
        output_data = list(reader)

    # Read the labels file
    with open(labels_file_path, 'r') as file:
        reader = csv.reader(file)
        labels_data = list(reader)

    # Iterate over each row of output
    for i in range(len(output_data)):
        output_row = output_data[i]
        labels_row = labels_data[i]

        for j in range(0, len(output_row), 2):
            candidate = output_row[j]
            probability = float(output_row[j + 1])

            if candidate in labels_row and probability > threshold:
                TP.extend([candidate, probability])
            if candidate not in labels_row and probability < threshold:
                TN.extend([candidate, probability])
            if candidate not in labels_row and probability > threshold:
                FP.extend([candidate, probability])
            if candidate in labels_row and probability < threshold:
                FN.extend([candidate, probability])

    TP_value = len(TP) / 2
    TN_value = len(TN) / 2
    FP_value = len(FP) / 2
    FN_value = len(FN) / 2
    TPR = (TP_value) / (TP_value + FN_value)
    TNR = (TN_value) / (TN_value + FP_value)
    ACC = (TP_value + TN_value) / (TP_value + TN_value + FP_value + FN_value)

    return TPR, TNR, ACC
