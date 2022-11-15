import os
import torch
import numpy as np
import cv2


def random_flip(x, label):
    prob_lr = np.random.random()
    if prob_lr > 0.5:
        x = np.fliplr(x).copy()
        label = np.fliplr(label).copy()
    return x, label


class ISPData(torch.utils.data.Dataset):
    def __init__(self, dataset_path=None, type="train"):
        super(ISPData, self).__init__()
        self.dataset_path = dataset_path
        file_set = []
        filenames = os.listdir(dataset_path)
        for file in filenames:
            if file[:-4] not in file_set:
                file_set.append(file[:-4])
        
        if type == "train":
            self.file_list = file_set[:round(len(file_set) * 0.8)]
        elif type == "val":
            self.file_list = file_set[round(len(file_set) * 0.8):]
    
    def img_init_propose(self, x):
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = np.array(x).astype('float32') / 255
        x = np.transpose(x, (2, 0, 1))
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        image_name = self.file_list[idx]
        rgb_image = cv2.imread(os.path.join(self.dataset_path, image_name) + ".jpg", cv2.IMREAD_COLOR)
        raw_image = cv2.imread(os.path.join(self.dataset_path, image_name) + ".npy", cv2.IMREAD_COLOR)
        rgb_image = self.img_init_propose(rgb_image)
        raw_image = self.img_init_propose(raw_image)
        rgb_image, raw_image = random_flip(rgb_image, raw_image)
        rgb_image = torch.from_numpy(rgb_image).type(torch.FloatTensor)
        raw_image = torch.from_numpy(raw_image).type(torch.FloatTensor)
        return rgb_image, raw_image





