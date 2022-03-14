import torch
import torchvision
from PIL import Image
import os

class Loader(torch.utils.data.Dataset):
    def __init__(self, data_dir, state, data_aug = False):
        self.dir = data_dir
        self.fileList = [f[:-4] for f in os.listdir(self.dir) if os.path.isfile(os.path.join(self.dir, f)) and ".jpg" in f]
        self.indexes = range(len(self.fileList))
        self.dic = {k: self.fileList[k] for k in self.indexes}
        self.state = state
        self.resize = 512
        self.data_aug = data_aug

    def __len__(self):
        return len(self.fileList)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx = self.indexes[idx]
        img_name = os.path.join(self.dir, self.dic[idx] + ".jpg")
        image = Image.open(img_name).convert('L')
        image.verify()

        if self.data_aug == True and self.state == "train":
            mirror = torch.rand((2,))
            if mirror[0] >= 0.5 :
                image = torchvision.transforms.functional.hflip(image)
            if mirror[1] >= 0.5 : 
                image = torchvision.transforms.functional.vflip(image)
        image = torchvision.transforms.functional.resize(
            image, size=[self.resize, self.resize])
        image = (torchvision.transforms.functional.to_tensor(image))

        return image, self.dic[idx]
