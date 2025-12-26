import os
import torch
import decord
import random
from torch.utils.data import Dataset

class TemporalDataset(Dataset):
    def __init__(self, list_path):
        with open(list_path, "r", encoding="utf-8") as file:
            files = file.readlines()
        self.files = [file.strip() for file in files]
        self.resize, self.crop_size = 256, (256, 256)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        patient = self.files[idx]
        if "HER2" in patient:
            class_id = 0
        elif "TNBC" in patient:
            class_id = 1
        elif "Luminal" in patient:
            class_id = 2
        contrast_path = os.path.join(patient, "contrast.wmv")
        gray_path = contrast_path.replace("contrast", "gray")

        contrast_vid = decord.VideoReader(contrast_path)
        h, w = contrast_vid[0].shape[:2]

        scale = float(self.resize) / float(min(h, w))
        size = int(h * scale + 0.5), int(w * scale + 0.5)

        y = random.randint(0, size[0] - self.crop_size[0])
        x = random.randint(0, size[1] - self.crop_size[1])

        contrast_vid = decord.VideoReader(contrast_path, height=size[0], width=size[1])
        gray_vid = decord.VideoReader(gray_path, height=size[0], width=size[1])
        assert len(vid) == len(gray_vid)

        gray_frame = gray_vid[0].asnumpy()
        contrast_vid = vid.asnumpy()
        
        contrast_vid = contrast_vid[:, y : y + self.crop_size[0], x : x + self.crop_size[1]]
        gray_frame = gray_frame[y : y + self.crop_size[0], x : x + self.crop_size[1]]
    
        contrast_vid = torch.as_tensor(contrast_vid.transpose((3, 0, 1, 2)).copy())
        contrast_vid = contrast_vid.sub(127.5).div(127.5)
        gray_frame = torch.as_tensor(gray_frame.transpose((2, 0, 1)).copy())
        gray_frame = gray_frame.sub(127.5).div(127.5)

        return contrast_vid, gray_frame, class_id
