import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import time as time
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

# DATA PREPARE
batch_size = 512
new_image_size = 128

img_transform = transforms.Compose([
    transforms.CenterCrop(new_image_size),
    transforms.ToTensor()
])

celebA_folder = 'C:/Users/Mucahid/Desktop/img_align_celeba/'
dataset = dset.ImageFolder(root=celebA_folder, transform=img_transform)
lengths = [int(len(dataset) * 0.9), int(len(dataset) * 0.1) + 1]
# lengths = [182339, 20260]
train_set, test_set = torch.utils.data.random_split(dataset, lengths)
tr_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
tt_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.enConv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=(2, 1))

        self.enConv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2)

        self.enConv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2)

        self.enConv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2)

        self.midConv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        # self.BnNorm = nn.BatchNorm2d(1024)

        self.midConv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.decConv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)

        self.decConv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)

        self.decConv4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)

        self.decConv5 = nn.Conv2d(in_channels=16, out_channels=6, kernel_size=3, padding=1)

        self.out = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, tensor):
        tensor = self.enConv1(tensor)
        tensor = func.relu(tensor)

        tensor = self.enConv2(tensor)
        tensor = func.relu(tensor)

        tensor = self.enConv3(tensor)
        tensor = func.relu(tensor)

        tensor = self.enConv4(tensor)
        tensor = func.relu(tensor)

        tensor = self.midConv1(tensor)
        tensor = func.relu(tensor)

        # tensor = self.BnNorm(tensor)

        tensor = self.midConv2(tensor)
        tensor = func.relu(tensor)

        tensor = self.up(tensor)
        tensor = self.decConv2(tensor)
        tensor = func.relu(tensor)

        tensor = self.up(tensor)
        tensor = self.decConv3(tensor)
        tensor = func.relu(tensor)

        tensor = self.up(tensor)
        tensor = self.decConv4(tensor)
        tensor = func.relu(tensor)

        tensor = self.up(tensor)
        tensor = self.decConv5(tensor)
        tensor = func.relu(tensor)

        tensor = self.out(tensor)
        return tensor


device = torch.device('cuda')

model = NeuralNetwork()

model.train()
# model.eval()
model = model.to(device)
print(model)
l_R = 1e-4
epoch_size = 520
start = time.time()

optimizer = optim.Adam(model.parameters(), lr=l_R, weight_decay=1e-5)

criterion = nn.MSELoss()

for epoch in range(epoch_size):
    i = 0
    for i_batch, (image, _) in enumerate(tr_dataloader):
        labels = image.to(device)
        images = image.permute(3, 1, 2, 0)[:64].permute(3, 1, 2, 0).to(device)
        res = model(images)
        loss = criterion(res, labels)
        loss.backward()
        optimizer.step()
        print("Batch : ", i_batch, " Epoch : ", epoch, " Loss : ", loss.item())
        i += loss.item()
        if i_batch % 50 == 0:
            trans = transforms.ToPILImage()
            im = res.cpu()
            im_in = images.cpu()
            im_l = labels.cpu()
            im1 = trans(im[0])
            im2 = trans(im_l[0])
            im3 = trans(im_in[0])
            Image._show(im1)
            Image._show(im2)
            Image._show(im3)
    print(epoch, " epoch loss : ", i/357)

print("Time Spend =", time.time() - start)
