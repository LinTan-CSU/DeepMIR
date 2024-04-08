import optuna
import torch
from pre_training import Model
from Dataset import Dataset
import torch.utils.data as data
import torch.nn as nn

# Define the objective function for Optuna
def objective(trial):
    # Sample hyperparameters
    batch_size = trial.suggest_categorical('batch_size', [64, 100])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-3, 1e-2, step=1e-3)

    train_dataset = Dataset('../data/traindataset.npy',
                            '../data/trainlabels.npy')
    train_dataloader = data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    valid_dataset = Dataset('../data/validdataset.npy',
                            '../data/validlabels.npy')
    valid_dataloader = data.DataLoader(valid_dataset, shuffle=False, batch_size=batch_size)

    model = Model()
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    criterion = nn.BCELoss()

    # Initialize variables for early stopping
    best_loss = float('inf')
    patience = 5  # Number of epochs to wait for improvement
    counter = 0  # Counter for tracking epochs without improvement
    stopped_epoch = 0  # Variable to store the epoch when training stopped

    for epoch in range(100):

        for i, train_data in enumerate(train_dataloader, 0):
            inputs, labels = train_data
            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels).cuda()
            loss.backward()
            optimizer.step()

        # valid
        v_loss = 0.0
        valid_correct = 0
        model.eval()
        with torch.no_grad():
            for i, valid_data in enumerate(valid_dataloader, 0):
                inputs, labels = valid_data
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, labels).cuda()
                v_loss += loss.item()

                outputs[outputs >= 0.5] = 1
                outputs[outputs < 0.5] = 0
                num_correct = (outputs == labels).sum().item()
                acc = num_correct / inputs.shape[0]
                valid_correct += acc

            loss = v_loss / len(valid_dataloader)
            accuracy = valid_correct / len(valid_dataloader)

        # Check for early stopping
        if loss < best_loss:
            best_loss = loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                stopped_epoch = epoch
                print(f"Training stopped at epoch {stopped_epoch}.")
                break

    return accuracy

# Create an Optuna study and optimize the objective function
study = optuna.create_study(study_name='trial',direction='maximize',storage='sqlite:///trial.sqlite3')
study.optimize(objective, n_trials=20)

# Print the best hyperparameters and the best score
best_params = study.best_params
best_accuracy = study.best_value
print("Best Hyperparameters:", best_params)
print("Best Validation accuracy:", best_accuracy)

df = study.trials_dataframe()
print(df)
df.to_csv(r'./optuna_trials.csv', index=False)
