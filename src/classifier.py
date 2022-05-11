import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from tqdm.notebook import tqdm


# Define dataloader class
class ClassifierDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

# Read input
df = pd.read_csv("/users/home/felpac/dim_ancestry/ancestry_vae/results/22_03_balanced_batch20_zdims100_train0.8_hidden1000x3_test/data/train_latent_space_epoch50.csv", index_col=0)
metadata = pd.read_csv("/users/data/pr_00006/CHB_DBDS/metadata/metadata_clear_COVID.tsv", sep="\t")[["id", "ancestry"]]
df = df.rename(columns={'label': 'id'})

# Merge data and metadata
df =  pd.merge(df, metadata, on ='id')


# Separate data and targets
X = df.iloc[:, :-3].to_numpy()
print(df.iloc[:, :-3].columns.values)
init_targets = df.ancestry.to_numpy()

le = preprocessing.LabelEncoder()
le.fit(np.unique(init_targets))
y = le.transform(init_targets)

#ohe = OneHotEncoder(categories='auto', sparse=False)
#y = ohe.fit_transform(init_targets[:, np.newaxis])
#y_init =  ohe.inverse_transform(y).ravel()


# Split train-test and define dataloaders
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=init_targets, random_state=21)
train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

# Define class weights

#classes, class_counts = np.unique(ohe.inverse_transform(y_train).ravel(), return_counts=True)
#class_weights = 1./torch.tensor(class_counts, dtype=torch.float) 

#class_weights_all = class_weights[y_train]

"""
weighted_sampler = WeightedRandomSampler(
    weights=class_weights_all,
    num_samples=len(class_weights_all),
    replacement=True
)
"""
EPOCHS = 300
BATCH_SIZE = 20
LEARNING_RATE = 0.001
NUM_FEATURES = 2
NUM_CLASSES = len(np.unique(init_targets))
"""
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          sampler=weighted_sampler
)
"""

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE
)


test_loader = DataLoader(dataset=test_dataset, batch_size=1)


class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()
        
        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

model = MulticlassClassification(num_feature =NUM_FEATURES, num_class=NUM_CLASSES)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc

accuracy_stats = {
    'train': [],
    "test": []
}
loss_stats = {
    'train': [],
    "test": []
}


print("Begin training.")
for e in tqdm(range(1, EPOCHS+1)):
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    model.train()

    for X_train_batch, y_train_batch in train_loader:

        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()
        
        y_train_pred = model(X_train_batch)
        #print(y_train_pred.shape)
        #print(y_train_batch.shape)
        train_loss = criterion(y_train_pred, y_train_batch.squeeze())
        train_acc = multi_acc(y_train_pred, y_train_batch)
        
        train_loss.backward()
        optimizer.step()
        
        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()
        
        
    # VALIDATION    
    with torch.no_grad():
        
        test_epoch_loss = 0
        test_epoch_acc = 0
        
        model.eval()
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
            
            y_test_pred = model(X_test_batch)
                        
            test_loss = criterion(y_test_pred, y_test_batch)
            test_acc = multi_acc(y_test_pred, y_test_batch)
            
            test_epoch_loss += test_loss.item()
            test_epoch_acc += test_acc.item()
            loss_stats['train'].append(train_epoch_loss/len(train_loader))
            loss_stats['test'].append(test_epoch_loss/len(test_loader))
            accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
            accuracy_stats['test'].append(test_epoch_acc/len(test_loader))
                              
    
    print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | test Loss: {test_epoch_loss/len(test_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| test Acc: {test_epoch_acc/len(test_loader):.3f}')