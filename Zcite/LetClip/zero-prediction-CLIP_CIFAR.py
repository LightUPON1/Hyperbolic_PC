import os
import clip
import torch
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

dataset_select = 'CIFAR100'
# Download the dataset
root = os.path.expanduser("~/../home/Dataset")

if dataset_select == 'CIFAR10':
    test = CIFAR10(root, download=True, train=False, transform=preprocess)
    # transform=transforms.Compose([
    #     transforms.Resize(224),
    #     transforms.ToTensor()
    # ])
elif dataset_select == 'CIFAR100':
    test = CIFAR100(root, download=True, train=False, transform=preprocess)

test_loader = DataLoader(test, batch_size=100, shuffle=False)
# Prepare the inputs
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in test.classes]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

acc1_ALL = 0
acc5_ALL = 0
for image_input, labels in tqdm(test_loader):
    image_input = image_input.to(device)
    labels = labels.to(device)
    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity.topk(1)
    # values, indices = similarity.topk(5)
    acc1, acc5 = accuracy(indices, labels, topk=(1, 5))
    # print(f"acc1:{acc1}%, acc5:{acc5}%")
    acc1_ALL += acc1
    acc5_ALL += acc5
acc1_ALL = acc1_ALL/len(test_loader)
acc5_ALL = acc5_ALL/len(test_loader)

print(f"{dataset_select}--acc1_ALL:{acc1_ALL} %, acc5_ALL:{acc5_ALL} %")
# # Print the result
# print("\nTop predictions:\n")
# for value, index in zip(values, indices):
#     print(f"{cifar100_testset.classes[index]:>16s}: {100 * value.item():.2f}%")

