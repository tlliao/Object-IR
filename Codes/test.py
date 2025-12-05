# coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import imageio
from dataset import *
import os
import numpy as np
import cv2
from ultralytics import YOLO
import torchvision.transforms as T
import torch.nn.functional as F
from vgg import VGG19
vgg = VGG19()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))


model_yolo = YOLO('/mnt/wangr/Object-IR/Codes/yolo11n.pt')
def draw_box(img, boxes, color_dict, save_path):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        color = color_dict[i]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{i}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite(save_path, img)



def distoration_loss(input_tensor, output_tensor):
    
    input_tensor = (input_tensor + 1) / 2
    output_tensor = (output_tensor + 1) / 2


    model_yolo = YOLO('/mnt/wangr/Object-IR/Codes/yolo11n.pt')
    results_img1 = model_yolo.predict(input_tensor, verbose=False)
    results_img2 = model_yolo.predict(output_tensor, verbose=False)

    batch_boxes1 = results_img1[0].boxes.xyxy
    batch_boxes2 = results_img2[0].boxes.xyxy
   
    
    if batch_boxes1.shape[0] == 0 or batch_boxes2.shape[0] == 0:
        print("no bouding")
        return 0
    
    def extract_pixels(image, box, target_size=(224, 224)):
       
        x1, y1, x2, y2 = map(int, box)
        cropped_region = image[:, y1:y2, x1:x2]  
      
        cropped_region = F.interpolate(cropped_region.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
        features = vgg(cropped_region)
        conv4_2_features = features['conv4_2'].squeeze(0)  # Shape: [C, H', W']
        # Flatten the feature map for similarity computation
        flattened_features = conv4_2_features.reshape(-1)
        return flattened_features  


   
    matched_pairs = []
    used_indices = set()
    similarity_threshold = 0.9
    for i, box1 in enumerate(batch_boxes1):
        best_match = None
        best_score = float('-inf') 

       
        box1_pixels = extract_pixels(input_tensor[0], box1)

        for j, box2 in enumerate(batch_boxes2):
            if j not in used_indices:
               
                box2_pixels = extract_pixels(output_tensor[0], box2)
                
               
                cosine_similarity = F.cosine_similarity(box1_pixels.unsqueeze(0), box2_pixels.unsqueeze(0), dim=1).item()
                
              
                if cosine_similarity > best_score:
                    best_score = cosine_similarity
                    best_match = j
        
      
        if best_match is not None and best_score >= similarity_threshold:
            matched_pairs.append((i, best_match))
            used_indices.add(best_match)
    if not matched_pairs:
        print("no matched")
        return 0
    
    
    color_dict = {i: (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for i in range(len(matched_pairs))}

    input_img1 = (input_tensor[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    input_img2 = (output_tensor[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    draw_box(input_img1.copy(), [batch_boxes1[i].cpu().numpy() for i, _ in matched_pairs], color_dict, "matched_boxes_img1.png")
    draw_box(input_img2.copy(), [batch_boxes2[j].cpu().numpy() for _, j in matched_pairs], color_dict, "matched_boxes_img2.png")
    
   
    distortion_loss = 0
    for i, j in matched_pairs:
        box1 = batch_boxes1[i]
        box2 = batch_boxes2[j]
        

       
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
        ar1 = w1 / h1
        ar2 = w2 / h2
       
        diff = torch.abs(ar1 - ar2) /ar1
   
        distortion_loss += diff


    distortion_loss /= len(matched_pairs)
    
    return distortion_loss
    
def test(args):
    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # dataset
    test_data = TestDataset(data_path=args.test_path, ratio=args.ratio)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=False)

    print("##################start testing#######################")
    dis_list = []
    

    for i, batch_value in enumerate(test_loader):

        input_tensor = batch_value[0].float()
        output_tensor = batch_value[1].float()

        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            output_tensor = output_tensor.cuda()

            dis = distoration_loss(input_tensor, output_tensor)
            
            if(dis != 0):
                print('i = {}, distoration_erro = {:.6f}'.format(i+1, dis))
                
                user_input = input("Enter 'y' to continue, 'n' to skip: ").strip().lower()                
                if user_input == 'y':
                    dis_list.append(dis)
            
            torch.cuda.empty_cache()
    
    print("=================== Analysis ==================")
    
    dis_list = [item.cpu().numpy() if isinstance(item, torch.Tensor) else item for item in dis_list]
    dis_array = np.array(dis_list)
    print('average distoration erro:', np.mean(dis_array))


    print("##################end testing#######################")


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--ratio', type=float, default=0.5)
    parser.add_argument('--test_path', type=str, default='/mnt/wangr/Object-IR')
    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
    test(args)
