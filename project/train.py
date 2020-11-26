# Adapted from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import time
import torchvision
import copy
import torch.optim as optim
import numpy as np

from models import initialize_model, SmallModel, eval_suite

# TODO transforms?
def get_dataloaders(dataset_name, root, batch_size):
    if dataset_name == 'mnist':
        dataset = torchvision.datasets.MNIST(root, train=True, download=True, 
                            transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                            ]))
        val_dataset = torchvision.datasets.MNIST(root, train=False, download=True,
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                            ]))

        print(dataset[0])

        train_data_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=True)

        val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)
        label_names = list(map(str, range(10)))

    elif dataset_name == "celeba":
        dataset = torchvision.datasets.CelebA(root, split='train', target_type='attr', download=True,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(
                                                (0.0,0.0,0.0), (1.0,1.0,1.0))
                                            ]))
        val_dataset = torchvision.datasets.CelebA(root, split='valid', target_type='attr', download=True,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(
                                                (0.0,0.0,0.0), (1.0,1.0,1.0))
                                            ]))

        train_data_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=True)

        val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)
        label_names = []

    elif dataset_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root, train=True, download=True,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(
                                                (0.0,0.0,0.0), (1.0,1.0,1.0))
                                            ]))
        val_dataset = torchvision.datasets.CIFAR10(root, train=False, download=True,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(
                                                (0.0,0.0,0.0), (1.0,1.0,1.0))
                                            ]))

        train_data_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=True)

        val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)
        label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    else:
        print("Invalid dataset name, exiting...")
        exit()

    return {'train': train_data_loader, 'val': val_data_loader, 'label_names': label_names}


def train_model(model, num_classes, dataloaders, criterion, optimizer, num_epochs=25):
    """Train a model and save best weights

    Args:
        model (nn.Module): Model to train
        dataloaders ([dataloader, dataloader]): Training dataloader and validation dataloader
        criterion (function): Loss function
        optimizer (torch.optim): Optimizer for training
        num_epochs (int, optional): Number of epochs to train for. Defaults to 25.

    Returns:
        (model, validation_accuracy): Model with best weights, Array of validation loss over training
    """
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            if phase == 'val':
                predicts_history = []
                ys_history = []

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    # print(loss.item())

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if phase == 'val':
                    predicts_history.extend(preds.clone().detach().cpu().numpy().tolist())
                    ys_history.extend(labels.clone().detach().cpu().numpy().tolist())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val': 
                print(eval_suite(np.array(predicts_history), np.array(ys_history), dataloaders['label_names']))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model
    model_name = 'resnet'
    num_classes = 10
    feature_extract = False
    # model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    model_ft = SmallModel(num_classes, 32, 32, 3)

    # Print the model we just instantiated
    print(model_ft)

    # Data Loaders
    data_loaders = get_dataloaders('cifar10', 'project/data', 32)
    # Train
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    train_model(model_ft, num_classes, data_loaders, criterion, optimizer)
