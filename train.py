import torchvision
import torch.nn as nn
from torch.utils.data import Dataset
import torch
import sklearn.metrics
import torch.optim as optim
from torchvision import transforms
from dataset import RetinopathyDatasetTrain
from tqdm import tqdm_notebook as tqdm
from torchvision import transforms
from torch.optim import lr_scheduler
import time
from model import DRModel
import pandas as pd
import os
from rounder import rounder
import wandb

wandb.init(project='aptos')
config = wandb.config
config.learning_rate = 2e-3
config.schedulerstep= 10
config.optimizer = 'Adam'
config.model = 'effnetb3'
config.epochs = 10


device = torch.device("cuda:0")
train_dataset = RetinopathyDatasetTrain(csv_file='../content/train_split.csv',transform = None)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_dataset = RetinopathyDatasetTrain(csv_file='../content/val_split.csv',transform = None)
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4)
model = DRModel(arch='efficientnet-b3')
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-3)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10)


wandb.watch(model)
since = time.time()
criterion = nn.SmoothL1Loss()
num_epochs = 10
losses = []
epochs =[]
acc = []
org_labels = []
org_out = []
for epoch in  range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    scheduler.step()
    model.train()
    running_loss = 0.0
    tk0 = tqdm(train_data_loader, total=int(len(train_data_loader)))
    counter = 0
    for bi, d in enumerate(tk0):
        inputs = d["image"]
        labels = d["labels"].view(-1, 1)
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        org_out.extend(rounder(outputs))
        org_labels.extend(labels.view(-1).cpu().detach().numpy())
        running_loss += loss.item() 
        counter += 1
        tk0.set_postfix(loss=(running_loss / (counter * train_data_loader.batch_size)))
    epoch_loss = running_loss / len(train_data_loader)
    acc.append(sklearn.metrics.accuracy_score(org_labels , org_out))
    losses.append(epoch_loss)
    epochs.append(epoch)
    print('Training Loss: {:.4f}'.format(epoch_loss))
    wandb.log({"Train loss": epoch_loss})   
    avg_val_loss = 0.0
    model.eval()
    tk1 = tqdm(val_data_loader, total=int(len(val_data_loader)))
    with torch.no_grad():
        for bi,d in enumerate(tk1):
            inputs = d["image"]
            labels = d["labels"].view(-1, 1)
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            outputs = model(inputs)
            avg_val_loss += criterion(outputs, labels).item() / len(val_data_loader)
        
    print('Validation Loss: {:.4f}'.format(avg_val_loss))
    wandb.log({"Validation loss": avg_val_loss})


time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
torch.save(model.state_dict(), "effnet_baseline.pth")