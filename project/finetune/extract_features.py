from cifar10_module import get_classifier
from torchvision import datasets, transforms
from data import load_dataset
import torch
import pickle as pkl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model/data loader
model_ft = get_classifier('resnet18', pretrained=True)
model_ft.eval()
train_dataset, val_dataset, label_names = load_dataset('cifar10')

train_dat = []
i = 0
for x, y in train_dataset:
	features = model_ft(x.unsqueeze(0)).detach().cpu().numpy()
	train_dat.append({'y': y, 'features': features})
	print(i)
	i += 1

val_dat = []
for x, y in val_dataset:
	features = model_ft(x.unsqueeze(0)).detach().cpu().numpy()
	val_dat.append({'y': y, 'features': features})
	print(i)
	i += 1

with open('./train_features.pkl', 'wb') as f:
	pkl.dump(train_dat, f)

with open('./val_features.pkl', 'wb') as f:
	pkl.dump(val_dat, f)


