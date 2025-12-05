from torch.utils.data import Dataset
import  numpy as np
import cv2, torch
import os
import glob
from collections import OrderedDict
import random
import torch.nn.functional as F


class TrainDataset(Dataset):
    def __init__(self, data_path):

        self.width = 224
        self.height = 224
       
        self.train_path = data_path
        self.datas = OrderedDict()
        
        datas = glob.glob(os.path.join(self.train_path, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'input':
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['image'].sort()
       

    def __getitem__(self, index):
        
        # load image
        
        input = cv2.imread(self.datas['input']['image'][index])
        input = cv2.resize(input, (self.width, self.height))
        input = input.astype(dtype=np.float32) 
        input = (input / 127.5) - 1.0  
        input = np.transpose(input, [2, 0, 1])
       
        # convert to tensor
        input_tensor = torch.tensor(input)
      
        
        return input_tensor, input_tensor
       

    def __len__(self):

        return len(self.datas['input']['image'])#number of images in 'input'

class TestDataset(Dataset):
    def __init__(self, data_path, ratio=0.5):
        self.train_path = data_path
        self.ratio = ratio  # Store the ratio parameter
        self.datas = OrderedDict()
        
        datas = glob.glob(os.path.join(self.train_path, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'input' or data_name == 'output':
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['image'].sort()
       
    def __getitem__(self, index):
        # Load image
        input_img = cv2.imread(self.datas['input']['image'][index])       
        input_img = input_img.astype(dtype=np.float32)  
        input_img = (input_img / 127.5) - 1.0
        
        # Resize input to 640x640
        input_img = cv2.resize(input_img, (640, 640), interpolation=cv2.INTER_LINEAR)
        input_img = np.transpose(input_img, [2, 0, 1])  # Shape: [C, H, W]
        
        output_img = cv2.imread(self.datas['output']['image'][index])        
        output_img = output_img.astype(dtype=np.float32) 
        output_img = (output_img / 127.5) - 1.0
        
        # Resize output to width = 640 * ratio, height = 640
        output_width = int(640 * self.ratio)
        output_img = cv2.resize(output_img, (output_width, 640), interpolation=cv2.INTER_LINEAR)
        output_img = np.transpose(output_img, [2, 0, 1])  # Shape: [C, H, W]
        
        # Convert to tensor
        input_tensor = torch.tensor(input_img)
        output_tensor = torch.tensor(output_img)
        
        return input_tensor, output_tensor
    
    def __len__(self):
        return len(self.datas['input']['image'])
       

    def __len__(self):

        return len(self.datas['input']['image'])

class OutDataset(Dataset):
    def __init__(self, data_path):

        
        self.test_path = data_path
        self.datas = OrderedDict()
        
        datas = glob.glob(os.path.join(self.test_path, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'input':
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.[jp][pn]g'))
                self.datas[data_name]['image'].sort()
        #print(self.datas.keys())

    def __getitem__(self, index):
        
        # load image1
        input = cv2.imread(self.datas['input']['image'][index])
        input = input.astype(dtype=np.float32)
        input = (input / 127.5) - 1.0
        input = np.transpose(input, [2, 0, 1])       
        # convert to tensor
        input_tensor = torch.tensor(input)
       
        return (input_tensor, input_tensor)

    def __len__(self):

        return len(self.datas['input']['image'])


def main():
    data_path = "/mnt/wangr/Date/train"  
    dataset = TrainDataset(data_path)

    print(f"Total images in dataset: {len(dataset)}")

    failed_images = []
    for index in range(len(dataset)):
        try:
            _ = dataset[index] 
        except ValueError as e:
            print(e)  
            failed_images.append(str(e).split(": ")[-1])  


    if failed_images:
        print("\nFailed to load the following images:")
        for file in failed_images:
            print(file)
    else:
        print("\nAll images loaded successfully!")

if __name__ == "__main__":
    main()

