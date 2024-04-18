import numpy as np
from neural_nets import GRUClassifier
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F

# TODO: disregard count=0 labels

# print parameters
show_output = True

# load numpy data
data = np.load('dataset_not_normalized.npz', allow_pickle=True)
features = data['array1']
labels = data['array2']

# transform numpy arrays to tensors
features = torch.from_numpy(features).float()
labels = torch.from_numpy(labels)

if show_output:
    print(features.shape)
    print(labels.shape)

# parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = features.shape[2]
hidden_size = 50
num_layers = 2
num_classes = 3
num_epochs = 10
learning_rate = 0.01
batch_size = 1
num_samples = features.shape[0]
split_size = 0.9  # 0.8 = 80% train & 20% test

# create TensorDataset
dataset = TensorDataset(features, labels)

# Define sizes for splitting
train_size = int(split_size * len(dataset))
test_size = len(dataset) - train_size

# split dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# create model and train it
model = GRUClassifier(input_size, hidden_size, num_layers, num_classes)
model.train_model(train_loader, num_epochs, learning_rate, device, test_loader)

# test on one feature
print(f"true label: {labels[0]}")
print(f"predicted label: {torch.argmax(F.softmax(model.forward(x=features[0].unsqueeze(0)), dim=1), dim=1)}")
