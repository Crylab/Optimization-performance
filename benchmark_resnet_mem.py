import psutil
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
from itertools import islice
from torch.profiler import profile, record_function, ProfilerActivity


def get_memory_usage():
    """
    Returns the current Resident Set Size (RSS) memory usage of the process.
    """
    # Get the ID of the current process
    pid = os.getpid()
    process = psutil.Process(pid)
    
    # Get memory info
    memory_info = process.memory_info()
    
    # memory_info.rss is in bytes. Divide by 1024^2 to get Megabytes (MB)
    rss_mb = memory_info.rss / (1024 ** 2)
    
    return rss_mb



def train_resnet_cifar10(command):

    if os.path.isfile(f'data_ResNet_mem/ResNet_{command}.json'):
        print('Training was preempted')
        return
    else:
        print(f'Starting training of {command}')


    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    num_epochs = 5
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
    #optimization_loss = []

    # Training loop
    # for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    memory_usage_list = []
    
    for inputs, labels in islice(train_loader, 100):  # Limiting to first 100 batches for memory benchmarking
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        memory_usage_list.append(get_memory_usage())
        optimizer.step()

        running_loss += loss.item()

    return np.average(memory_usage_list)

if __name__ == "__main__":
    is_parallel = False
    folder_name = "data_ResNet_mem"

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
        learning_rate = 0.01
        if "torch.optim.AMSgrad" == Algorithm:
            result = "torch.optim.Adam(model.parameters(), lr=" + str(learning_rate) + ", amsgrad=True)"
        else:
            result = Algorithm + "(model.parameters(), lr=" + str(learning_rate) + ")"
        task_list.append((result, ))

    overall_memory_usage = []
    
    for task in task_list:
        memory_usage = train_resnet_cifar10(*task)
        overall_memory_usage.append(memory_usage)
        print(f"Memory usage for {task}: {memory_usage} MB")

    with open(f'data_ResNet_mem/ResNet_memory.json', "w") as json_file:
        json.dump(overall_memory_usage, json_file, indent=4)