import torch
import torch.nn as nn
import torch_optimizer as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from multiprocessing import Pool
import multiprocessing
import json
import os.path
import os


def train_resnet_cifar10(command):

    if os.path.isfile(f'data_ResNet/ResNet_{command}.json'):
        print('Training was preempted')
        return
    else:
        print(f'Starting training of {command}')


    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    num_epochs = 100
    batch_size = 128

    # Transformations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    # Load ResNet18 model
    model = models.resnet18(pretrained=False, num_classes=10)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = eval(command)
    optimization_loss = []
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        Average_loss_per_batch = running_loss/len(train_loader)
        optimization_loss.append(Average_loss_per_batch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {Average_loss_per_batch:.4f}")

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    Total_accuracy = 100 * correct / total
    print(f"Test Accuracy: {Total_accuracy:.2f}%")

    output_data = {
        "Total_accuracy": Total_accuracy,
        "Optimization_path": optimization_loss
    }

    with open(f'data_ResNet/ResNet_{command}.json', "w") as json_file:
        json.dump(output_data, json_file, indent=4)

if __name__ == "__main__":
    is_parallel = False
    folder_name = "data_ResNet"

    # Check and create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")
    algorithm_list = [
        "torch.optim.SGD",
        "torch.optim.Adagrad",
        "optim.AggMo",
        "torch.optim.Rprop",
        "torch.optim.RMSprop",
        "torch.optim.Adam",
        "torch.optim.Adamax",
        "torch.optim.NAdam",
        "torch.optim.RAdam",
        "torch.optim.AMSgrad", 
        "optim.NovoGrad",
        "optim.SWATS",
        "optim.DiffGrad", 
        "optim.Yogi",
        "optim.Lamb",
        "optim.AdamP",
        "torch.optim.AdamW",
        "optim.AdaMod",
        "optim.MADGRAD",
        "optim.AdaBound", 
        "optim.PID",
        "optim.QHAdam",
    ]

    task_list = []
    for Algorithm in algorithm_list:
        for learning_rate in [10**i for i in range(-3, -1)]:
            if "torch.optim.AMSgrad" == Algorithm:
                result = "torch.optim.Adam(model.parameters(), lr=" + str(learning_rate) + ", amsgrad=True)"
            else:
                result = Algorithm + "(model.parameters(), lr=" + str(learning_rate) + ")"
            task_list.append((result, ))
    
    if is_parallel:
        with Pool(processes=3) as pool:
            # Use starmap to pass multiple arguments
            results = pool.starmap(train_resnet_cifar10, task_list, )
    else:
        for task in task_list:
            train_resnet_cifar10(*task)
