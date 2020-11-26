from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def preprocess(img_size=224):
    img_preprocess = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return img_preprocess


def load_dataset(dataset_name, img_size, root='data'):
    img_preprocess = preprocess(img_size)

    if dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(root, train=True, transform=img_preprocess, download=True)
        val_dataset = datasets.CIFAR10(root, train=False, transform=img_preprocess, download=True)

    elif dataset_name == "celeba":
        # Note: download=True fails when daily quota on this dataset has been reached
        train_dataset = datasets.CelebA(root, split='train', target_type='attr',
                                        transform=img_preprocess,
                                        download=True)
        val_dataset = datasets.CelebA(root, split='valid', target_type='attr',
                                      transform=img_preprocess,
                                      download=True)

    else:
        print("Invalid dataset name, exiting...")
        return

    return train_dataset, val_dataset


def get_dataloader(train_dataset, val_dataset, batch_size):
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_data_loader, val_data_loader
