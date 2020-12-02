from torchvision import datasets, transforms


def preprocess(dataset_name):
    if dataset_name == 'mnist':
        img_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif dataset_name == 'celeba' or dataset_name == 'cifar10':
        img_preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
    else:
        print("Invalid dataset name, exiting...")
        return

    return img_preprocess


def load_dataset(dataset_name, root='data'):
    img_preprocess = preprocess(dataset_name)

    if dataset_name == "mnist":
        train_dataset = datasets.MNIST(root, train=True, download=True, transform=img_preprocess)
        val_dataset = datasets.MNIST(root, train=False, download=True, transform=img_preprocess)
        label_names = list(map(str, range(10)))

    elif dataset_name == "celeba":
        # Note: download=True fails when daily quota on this dataset has been reached
        train_dataset = datasets.CelebA(root, split='train', target_type='attr',
                                        transform=img_preprocess,
                                        download=True)
        val_dataset = datasets.CelebA(root, split='valid', target_type='attr',
                                      transform=img_preprocess,
                                      download=True)
        label_names = []

    elif dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(root, train=True, download=True, transform=img_preprocess)
        val_dataset = datasets.CIFAR10(root, train=False, download=True, transform=img_preprocess)
        label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    else:
        print("Invalid dataset name, exiting...")
        return

    return train_dataset, val_dataset, label_names
