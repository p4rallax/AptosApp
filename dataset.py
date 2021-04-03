import torchvision
import torch.nn as nn
from torch.utils.data import Dataset
import torch
import sklearn.metrics
import torch.optim as optim
from torchvision import transforms
from PIL import Image, ImageFile
import pandas as pd
import os
import numpy as np

class RetinopathyDatasetTrain(Dataset):

    def __init__(self, csv_file , transform):

        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join('../content/train/train_images', self.data.loc[idx, 'id_code'] + '.png')
        image = Image.open(img_name)
        image = image.resize((224, 224), resample=Image.BILINEAR)
        image = np.asarray( image, dtype=np.uint8 )
        if self.transform:
            result = self.transform(image=image)
            image = result['image']
        label = torch.tensor(self.data.loc[idx, 'diagnosis'])
        return {'image': transforms.ToTensor()(image),
                'labels': label
                }