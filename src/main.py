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

start = time.time()
train_dataset = Dataset('../data/traindataset.npy', '../data/trainlabels.npy')
train_dataloader = data.DataLoader(train_dataset, shuffle=True, batch_size=64)
valid_dataset = Dataset('../data/validdataset.npy', '../data/validlabels.npy')
valid_dataloader = data.DataLoader(valid_dataset, shuffle=False, batch_size=64)


model = Model()
model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.004)


criterion = nn.BCELoss()
epochs = 100
batch_size = 64

# create a timestamp string to use in the log filename
timestamp = time.strftime("%Y%m%d-%H%M%S")

# configure the logging module
logging.basicConfig(filename=f'../log/DeepMIR_training_{timestamp}.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

state_dict = {
    'batch_size': batch_size,
    'epoch': epochs,
    'optimizer_state_dict': optimizer.state_dict()
        }
logging.info(state_dict)

# close the log file
logging.shutdown()
now = datetime.datetime.now()
train_loss = []
train_accuracy = []
valid_loss = []
valid_accuracy = []

# Initialize variables for early stopping
best_loss = float('inf')
patience = 5  # Number of epochs to wait for improvement
counter = 0  # Counter for tracking epochs without improvement
stopped_epoch = 0  # Variable to store the epoch when training stopped

print('start training')


for epoch in range(epochs):
    # train
    t_loss = 0.0
    train_correct = 0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels).cuda()
        loss.backward()
        optimizer.step()
        t_loss += loss.item()
        if i % 100 == 99:
            print(f"Epoch {epoch + 1}, train loss: {t_loss / (i + 1)}")

        outputs[outputs >= 0.5] = 1
        outputs[outputs < 0.5] = 0
        num_correct = (outputs == labels).sum().item()
        acc = num_correct / inputs.shape[0]
        train_correct += acc


    loss = t_loss / len(train_dataloader)
    accuracy = train_correct / len(train_dataloader)
    print(f"train loss: {loss}")
    print(f"train accuracy: {accuracy}")
    train_loss.append(loss)
    train_accuracy.append(accuracy)

    # valid
    v_loss = 0.0
    valid_correct = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(valid_dataloader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels).cuda()
            v_loss += loss.item()
            if i % 100 == 99:
                print(f"Epoch {epoch + 1}, valid loss: {v_loss / (i + 1)}")

            outputs[outputs >= 0.5] = 1
            outputs[outputs < 0.5] = 0
            num_correct = (outputs == labels).sum().item()
            acc = num_correct / inputs.shape[0]
            valid_correct += acc

        loss = v_loss / len(valid_dataloader)
        accuracy = valid_correct / len(valid_dataloader)
        print(f"valid loss: {loss}")
        print(f"valid accuracy: {accuracy}")
        valid_loss.append(loss)
        valid_accuracy.append(accuracy)

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


torch.save(model.state_dict(), '../model/DeepMIR_training_{}.pth'.format(now.strftime('%Y-%m-%d_%H-%M-%S')))

np.save('../res/DeepMIR_train_loss_{}.npy'.format(now.strftime('%Y-%m-%d_%H-%M-%S')), train_loss)
np.save('../res/DeepMIR_valid_loss_{}.npy'.format(now.strftime('%Y-%m-%d_%H-%M-%S')), valid_loss)
np.save('../res/DeepMIR_train_accuracy_{}.npy'.format(now.strftime('%Y-%m-%d_%H-%M-%S')), train_accuracy)
np.save('../res/DeepMIR_valid_accuracy_{}.npy'.format(now.strftime('%Y-%m-%d_%H-%M-%S')), valid_accuracy)

fig = plt.figure(figsize=(8, 4.5))
ax = fig.add_subplot(111)
ax.set_ylabel('Accuracy', size=15)
ax.set_xlabel('Epoch', size=15)
lns1 = ax.plot(train_accuracy, label='Acc_training', color='r')
lns2 = ax.plot(valid_accuracy, label='Acc_validation', color='g')
ax2 = ax.twinx()
ax2.set_ylabel('Loss', size=15)
lns3 = ax2.plot(train_loss, label='Loss_training', color='b')
lns4 = ax2.plot(valid_loss, label='Loss_validation', color='orange')
lns = lns1 + lns2 + lns3 + lns4
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=7)
plt.savefig('../fig/DeepMIR_training_{}.jpg'.format(now.strftime('%Y-%m-%d_%H-%M-%S')), dpi=700, bbox_inches='tight')
plt.show()

end = time.time()
print('Training finished, time:%.2fSeconds' % (end - start))
