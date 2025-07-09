import torch.nn as nn
import torch
from dataset import Dataset
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from DeepMIR import Model
import numpy as np

def train(
    name, lr=None, epochs=None, batch_size=None, patience=None, weight_decay=None
):
    """
    * Train the DeepMIR model with specified hyperparameters
    *
    * Parameters
    * ----------
    * name : str
    *     Name identifier for saving model and results.
    * lr : float
    *     Learning rate for the optimizer.
    * epochs : int
    *     Number of training epochs.
    * batch_size : int
    *     Batch size for training and validation.
    * patience : int
    *     Number of epochs to wait for improvement before early stopping.
    * weight_decay : float
    *     Weight decay (L2 regularization) for the optimizer.
    *
    * Returns
    * -------
    * None
    *     Saves the trained model and training/validation metrics to disk.
    """
    model = Model()
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    trainsetpath = '../data/train_set.npy'
    trainlabelspath = '../data/train_labels.npy'
    validsetpath = '../data/val_set.npy'
    validlabelspath = '../data/val_labels.npy'
    train_dataset = Dataset(trainsetpath, trainlabelspath)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    valid_dataset = Dataset(validsetpath, validlabelspath)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, shuffle=False, batch_size=batch_size)
    
    writer = SummaryWriter()          
    train_loss = []
    train_accuracy = []
    valid_loss = []
    valid_accuracy = []
    train_tpr = []
    train_tnr = []
    valid_tpr = []
    valid_tnr = []
          
    best_loss = float('inf')
    patience = patience
    counter = 0

    print('start training')

    for epoch in range(epochs):
        # train
        t_loss = 0.0
        train_correct = 0
        train_tp = 0
        train_tn = 0
        train_p = 0
        train_n = 0
        model.train()

        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            noise = torch.randn_like(inputs) * 0.01
            inputs = inputs + noise
            labels = labels.cuda()
            labels = labels.unsqueeze(1)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

            threshold = 0.5
            outputs[outputs >= threshold] = 1
            outputs[outputs < threshold] = 0
            train_tp += ((outputs == 1) & (labels == 1)).sum().item()
            train_tn += ((outputs == 0) & (labels == 0)).sum().item()
            train_p += (labels == 1).sum().item()
            train_n += (labels == 0).sum().item()

            num_correct = (outputs == labels).sum().item()
            acc = num_correct / (labels.shape[0] * labels.shape[1])
            train_correct += acc

        t_loss = t_loss / len(train_dataloader)
        t_acc = train_correct / len(train_dataloader)

        print(f"Epoch {epoch + 1}, Total Loss: {t_loss:.4f}, train Accuracy: {t_acc:.4f}")
        train_loss.append(t_loss)
        train_accuracy.append(t_acc)
        t_tpr = train_tp / train_p if train_p > 0 else 0
        t_tnr = train_tn / train_n if train_n > 0 else 0
        train_tpr.append(t_tpr)
        train_tnr.append(t_tnr)
        print(f"Train TPR: {t_tpr:.4f}, Train TNR: {t_tnr:.4f}")

        # valid
        v_loss = 0.0
        valid_correct = 0
        valid_tp = 0
        valid_tn = 0
        valid_p = 0
        valid_n = 0
        model.eval()

        with torch.no_grad():
            for i, data in enumerate(valid_dataloader, 0):
                inputs, labels = data
                inputs = inputs.cuda()
                # noise = torch.randn_like(inputs) * 0.05
                # inputs = inputs + noise
                labels = labels.cuda()
                labels = labels.unsqueeze(1)
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                v_loss += loss.item()
                threshold = 0.5
                outputs[outputs >= threshold] = 1
                outputs[outputs < threshold] = 0
                valid_tp += ((outputs == 1) & (labels == 1)).sum().item()
                valid_tn += ((outputs == 0) & (labels == 0)).sum().item()
                valid_p += (labels == 1).sum().item()
                valid_n += (labels == 0).sum().item()

                num_correct = (outputs == labels).sum().item()
                acc = num_correct / (labels.shape[0] * labels.shape[1])
                valid_correct += acc

            v_loss = v_loss / len(valid_dataloader)
            v_acc = valid_correct / len(valid_dataloader)

            scheduler.step(v_loss)
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning_rate', current_lr, epoch)

            print(f"Epoch {epoch + 1}, valid loss: {v_loss:.4f}, valid Accuracy: {v_acc:.4f}")

            valid_loss.append(v_loss)
            valid_accuracy.append(v_acc)
            v_tpr = valid_tp / valid_p if valid_p > 0 else 0
            v_tnr = valid_tn / valid_n if valid_n > 0 else 0
            valid_tpr.append(v_tpr)
            valid_tnr.append(v_tnr)
            print(f"Valid TPR: {v_tpr:.4f}, Valid TNR: {v_tnr:.4f}")

            writer.add_scalars('TPR/epoch', {
                'train': t_tpr,
                'val': v_tpr
            }, epoch)

            writer.add_scalars('TNR/epoch', {
                'train': t_tnr,
                'val': v_tnr
            }, epoch)

            writer.add_scalars('Loss/epoch', {
                'train': t_loss,
                'val': v_loss
            }, epoch)

            writer.add_scalars('Accuracy/epoch', {
                'train': t_acc,
                'val': v_acc
            }, epoch)

        if v_loss < best_loss:
            best_loss = v_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                stopped_epoch = epoch + 1
                print(f"Training stopped at epoch {stopped_epoch}.")
                break

    print('Training finished')
    writer.close()

    torch.save(model.state_dict(), '../model/train_{}.pth'.format(name))

    np.save('../res/train_loss_{}.npy'.format(name), train_loss)
    np.save('../res/valid_loss_{}.npy'.format(name), valid_loss)
    np.save('../res/train_accuracy_{}.npy'.format(name), train_accuracy)
    np.save('../res/valid_accuracy_{}.npy'.format(name), valid_accuracy)


    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    ax.set_ylabel('Accuracy', size=16, fontweight='bold')
    ax.set_xlabel('Epoch', size=16, fontweight='bold')
    lns1 = ax.plot(train_accuracy, label='Accuracy_training', color='r')
    lns2 = ax.plot(valid_accuracy, label='Accuracy_validation', color='g')
    ax2 = ax.twinx()
    ax2.set_ylabel('Loss', size=16, fontweight='bold')
    lns3 = ax2.plot(train_loss, label='Loss_training', color='b')
    lns4 = ax2.plot(valid_loss, label='Loss_validation', color='orange')
    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=7)
    plt.savefig('../fig/model/train_{}.jpg'.format(name), dpi=700, bbox_inches='tight')
    plt.show()    
          

if __name__ == '__main__':
    # replace with your data in the following code
    train(
        name='trial_1',
        lr=0.00823,
        epochs=100,
        batch_size=100,
        patience=5,
        weight_decay=1.04e-05
    )
