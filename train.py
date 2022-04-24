import torch 
import torchvision
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

from models import SimpleCNN

from filelock import FileLock
import os 
from tqdm import tqdm
import ray
import ray.tune as tune
from ray.tune.schedulers import AsyncHyperBandScheduler


# create dataloaders
def create_dataloader(path, transform, split = .1, batch_size = 64):
    """inputs a path to an image folder and returns train_loader and val_loader"""
    dataset = datasets.ImageFolder(root = path, transform = transform)

    # split data
    splits = [len(dataset) - int(len(dataset) * split), int(len(dataset) * split)]
    train_set, val_set = torch.utils.data.random_split(dataset, splits)

    # create loader
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = True)

    return train_loader, val_loader

def train_ray(config: dict) -> None:
    """create and train a model, config is of structure:
    config = {
        lr:
        l2:
        momentum:
        dropout_rate:
        image_size:
        epochs:
        """

    image_size = config['image_size']
    lr = config['lr']
    momentum = config['momentum']
    l2 = config['l2']
    dropout_rate = config['dropout_rate']
    epochs = config['epochs']

    # define device to use gpu if aviable 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create base transforms 
    base_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size),
        transforms.Normalize(mean = 0, std = 1),
    ])

    # load in data in a thread safe manor
    with FileLock(os.path.expanduser("~/data.lock")):
        train_loader, val_loader = create_dataloader(path = '~/data', transform = base_transforms)

    # create model 
    model = SimpleCNN(image_size = image_size, dropout_rate = dropout_rate).to(device)

    # create loss
    loss_fn = F.binary_cross_entropy
    
    # create optimizer 
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr = lr,
        momentum = momentum,
        weight_decay = l2)

    # iter through epochs 
    for epoch in range(epochs):
        train_loss = 0
        train_correct = 0

        # iter through training batches 
        for images, labels in train_loader:
            # reset optimizer
            optimizer.zero_grad()

            # load images
            images, labels = images.to(device), labels.to(device).float()

            # run images through model and calculate loss 
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # compute loss gradients and take a step
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # sum where outputs are equal to correct labels
            train_correct += (torch.round(outputs) == labels).float().mean().item()

        # average accuracy and loss over each batch
        train_accuracy = train_correct / len(train_loader)
        train_loss /= len(train_loader)

        val_loss = 0
        val_correct = 0

        # iter through val batches 
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()

                # run images through model and calculate loss 
                outputs = model(images)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item()

                val_correct += (torch.round(outputs) == labels).float().mean().item()

            # avergae accuracy and loss over epoch
            val_accuracy = val_correct / len(val_loader)
            val_loss /= len(val_loader)
        
        tune.report(
            train_accuracy = train_accuracy,
            train_loss = train_loss,
            val_accuracy = val_accuracy,
            val_loss = val_loss,
            epochs = epoch + 1,
            )


def hyp_search():
    # define scheduler 
    sched = AsyncHyperBandScheduler(metric = 'val_accuracy', mode = 'max')

    analysis = tune.run(
        train_ray,
        metric = "val_accuracy",
        mode = "max",
        name = 'exp',
        scheduler=sched,
        stop={
            "epochs": 4
        },
        resources_per_trial={"cpu": 4, "gpu": 1 if torch.cuda.is_available() else 0},  # set this for GPUs
        num_samples = 20,
        config = dict(
            image_size = 227 // 4,
            lr = tune.loguniform(1e-4, 1e-1),
            momentum = tune.uniform(0.1, 0.9),
            l2 = tune.loguniform(1e-5, 1e-2),
            dropout_rate = tune.uniform(.05, .35),
            epochs = 1000 # set to high number so iterations is used to stop not this
            )
    )

    print("Best config is:", analysis.best_config)

def train(image_size, lr, momentum, l2, dropout_rate, epochs):
    """create and train a model, config is of structure:
    config = {
        lr:
        l2:
        momentum:
        dropout_rate:
        image_size:
        epochs:
        """

    # define device to use gpu if aviable 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create base transforms 
    base_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size),
        transforms.Normalize(mean = 0, std = 1),
    ])

    # create training transforms
    train_tranforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(0, translate = (.1, .1), scale = (.9, 1.1))
    ])
    # load in data in a thread safe manor
    with FileLock(os.path.expanduser("~/data.lock")):
        train_loader, val_loader = create_dataloader(path = '/mnt/c/Users/14135/Desktop/Ray-Tune-Exp/data', transform = base_transforms)

    # create model 
    model = SimpleCNN(image_size = image_size, dropout_rate = dropout_rate).to(device)

    # create loss
    loss_fn = F.binary_cross_entropy
    
    # create optimizer 
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr = lr,
        momentum = momentum,
        weight_decay = l2)

    # iter through epochs 
    for epoch in range(epochs):
        train_loss = 0
        train_correct = 0

        # iter through training batches 
        for images, labels in tqdm(train_loader):
            # reset optimizer
            optimizer.zero_grad()

            # load images
            images, labels = images.to(device), labels.to(device).float()

            # apply training trainsforms 
            #images = train_tranforms(images)

            # run images through model and calculate loss 
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # compute loss gradients and take a step
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # sum where outputs are equal to correct labels
            train_correct += (torch.round(outputs) == labels).float().mean().item()

        # average accuracy and loss over each batch
        train_accuracy = train_correct / len(train_loader)
        train_loss /= len(train_loader)

        val_loss = 0
        val_correct = 0

        # iter through val batches 
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()

                # run images through model and calculate loss 
                outputs = model(images)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item()

                val_correct += (torch.round(outputs) == labels).float().mean().item()

        # avergae accuracy and loss over epoch
        val_accuracy = val_correct / len(val_loader)
        val_loss /= len(val_loader)

        print(f"Epoch {epoch} / {epochs}, {train_accuracy =}, {train_loss =}, {val_accuracy =}, {val_loss =}")

if __name__ == "__main__":
    hyp_search()





    
    
    

# 

