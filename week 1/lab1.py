import os
import getpass
import numpy as np
import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

# user defined variables
IMG_SIZE = 32
BATCH_SIZE = 16
DATASET_DIR_TRAIN = '/home/group09/code/Dataset_withVal/train'
DATASET_DIR_TEST = '/home/group09/code/Dataset_withVal/test'

# Other parameters
epochs = 800
log_interval = 10
writer = SummaryWriter('runs/test')

if not os.path.exists(DATASET_DIR_TRAIN):
    print('ERROR: train dataset directory ' + DATASET_DIR_TRAIN + ' do not exists!\n')
    quit()

if not os.path.exists(DATASET_DIR_TEST):
    print('ERROR: test dataset directory ' + DATASET_DIR_TEST + ' do not exists!\n')
    quit()

# setup cuda
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Loss
criterion = nn.CrossEntropyLoss()

# Define model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.conv4 = nn.Conv2d(64, 128, 1, stride=1)
        self.dropout1 = nn.Dropout(0.2)
        self.dense1 = nn.Linear(128, 8)
        self.softmax1 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = self.dropout1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x = self.dropout1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.dense1(x)
        output = self.softmax1(x)
        return output

# Preprocess the data: transform to tensor and apply image transforms
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomAffine(0, translate=(0.2, 0), scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomRotation(degrees=[0,5]),
    transforms.RandomHorizontalFlip(),
])

transform_test = transforms.Compose([transforms.ToTensor()])

trainset_ori = datasets.ImageFolder(DATASET_DIR_TRAIN, transform=transform_test)
trainset_daug = datasets.ImageFolder(DATASET_DIR_TRAIN, transform=transform_train)
testset = datasets.ImageFolder(DATASET_DIR_TEST, transform=transform_test)

# We use the original dataset plus the same dataset composed with transformed images
trainset = torch.utils.data.ConcatDataset([trainset_ori,trainset_daug])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

losses = []
accuracies = []


def train(model, device, trainloader, optimizer, epoch, running_loss, running_correct):
    model.train()
    loss_values = []
    n_total_steps = len(trainloader)
    count = 0
    for i, data in enumerate(trainloader, 0):
        count = count + 1
        input, labels = data
        input, labels = input.cuda(), labels.cuda()
        optimizer.zero_grad()
        output = cnn(input)
        loss = criterion(output, labels)
        loss_values.append(loss)
        loss.backward()
        optimizer.step()
        # For loss and accuracy visualisation
        running_loss += loss.item() * input.size(0)
        _, predicted = torch.max(output, 1)
        running_correct += torch.sum(predicted == labels.data)

    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_accuracy = running_correct.double() / (len(trainloader.dataset))
    writer.add_scalar('training loss', epoch_loss, epoch * n_total_steps + count)
    writer.add_scalar('training acc', epoch_accuracy, epoch * n_total_steps + count)
    losses.append(epoch_loss)
    accuracies.append(epoch_accuracy)
    print('Train Epoch: {}. Loss:{:.4f}, accuracy:{:.4f}'.format(epoch, epoch_loss, epoch_accuracy))

    return loss_values


cnn = CNN().to(device)

# Define optimiser
optimizer = optim.Adam(cnn.parameters(), lr=0.001)

running_loss = 0.0
running_correct = 0
for epoch in range(epochs):
    loss = train(cnn, device, trainloader, optimizer, epoch, running_loss, running_correct)


corrects = 0
accuracy = 0
totals = 807

# Test
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        outputs = cnn(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        true_pred = sum(c)
        corrects = corrects + true_pred

accuracy = corrects / totals
print('Accuracy = ' + str(accuracy))
