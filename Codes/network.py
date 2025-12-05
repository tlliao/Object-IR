import torch
import torch.nn as nn
import utils.torch_tps_transform as torch_tps_transform
import ssl
import cv2
import random
import numpy as np
from safetensors.torch import load_file
import predefine
import torchvision.transforms as T
import torch.nn.functional as F
import timm
import torchvision.models as models
from ultralytics import YOLO
import sys
import os
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resize_224 = T.Resize((224,224))

grid_h = predefine.GRID_H
grid_w = predefine.GRID_W


# draw mesh on image
# warp: h*w*3
# f_local: grid_h*grid_w*2
def draw_mesh_on_warp(warp, f_local):
    warp = np.ascontiguousarray(warp)

    point_color = (0, 0, 255) # BGR
    thickness = 2
    lineType = 8

    num = 1
    for i in range(grid_h+1):
        for j in range(grid_w+1):

            num = num + 1
            if j == grid_w and i == grid_h:
                continue
            elif j == grid_w:
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
            elif i == grid_h:
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)
            else :
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)

    return warp

# get rigid mesh
def get_rigid_mesh(batch_size, height, width):
    
    ww = torch.matmul(torch.ones([grid_h+1, 1]), torch.unsqueeze(torch.linspace(0., float(width), grid_w+1), 0))
    hh = torch.matmul(torch.unsqueeze(torch.linspace(0.0, float(height), grid_h+1), 1), torch.ones([1, grid_w+1]))
    if torch.cuda.is_available():
        ww = ww.cuda()
        hh = hh.cuda()

    ori_pt = torch.cat((ww.unsqueeze(2), hh.unsqueeze(2)),2) # (grid_h+1)*(grid_w+1)*2
    ori_pt = ori_pt.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return ori_pt

# normalize mesh from -1 ~ 1
def get_norm_mesh(mesh, height, width):
    mesh_w = mesh[..., 0] * 2. / float(width) - 1.
    mesh_h = mesh[..., 1] * 2. / float(height) - 1.
    norm_mesh = torch.stack([mesh_w, mesh_h], -1)  

    return norm_mesh
def get_stack_mesh(mesh):
    batch_size = mesh.size()[0]
    mesh_w = mesh[...,0]
    mesh_h = mesh[...,1]
    norm_mesh = torch.stack([mesh_w, mesh_h], 3) # bs*(grid_h+1)*(grid_w+1)*2

    return norm_mesh.reshape([batch_size, -1, 2]) # bs*-1*2

# random augmentation
def data_aug(img):
    # Randomly shift brightness
    random_brightness = torch.randn(1).uniform_(0.7,1.3).cuda()
    img_aug = img * random_brightness
    # Randomly shift color
    white = torch.ones([img.size()[0], img.size()[2], img.size()[3]]).cuda()
    random_colors = torch.randn(3).uniform_(0.7,1.3).cuda()
    color_image = torch.stack([white * random_colors[i] for i in range(3)], axis=1)
    img_aug  *= color_image

    # clip
    img_aug = torch.clamp(img_aug, -1, 1)

    return img_aug



def build_model(net, input_tensor, is_training):

    batch_size, _, img_h, img_w = input_tensor.size()

    if is_training == True:
        aug_input_tensor = data_aug(input_tensor)
        motion_primary= net(aug_input_tensor, is_training)
    else:
        motion_primary, motion_fine = net(input_tensor)
    
    min_scale = 0.25
    max_scale = 0.5
    scale_factor = random.uniform(min_scale, max_scale)

    out_h = img_h 
    out_w = int(img_w * scale_factor)

    ori_mesh = get_rigid_mesh(batch_size, img_h, img_w) 
    rigid_mesh = get_rigid_mesh(batch_size, out_h, out_w)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, out_h, out_w)
    
    mesh_pri = rigid_mesh + motion_primary
    norm_mesh_pri = get_norm_mesh(mesh_pri, out_h, out_w)
    
    stack_rigid_mesh = get_stack_mesh(norm_rigid_mesh)
    stack_mesh_pri = get_stack_mesh(norm_mesh_pri)
    

    mask = torch.ones_like(input_tensor)
    
    if torch.cuda.is_available():
        mask = mask.cuda()

    out_dict = {}
    out_tps_pri = torch_tps_transform.transformer(torch.cat((input_tensor, mask), 1), stack_mesh_pri, stack_rigid_mesh,  (out_h, out_w))
    warp_mesh_pri = out_tps_pri[:,0:3,...]   
    
    '''
    warp = ((warp_mesh_pri[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
    cv2.imwrite('output_image_rgb1.png', warp)
    '''
 
    out_dict.update(motion_primary=motion_primary, warp_primary = warp_mesh_pri, rigid_mesh = rigid_mesh, mesh_pri=mesh_pri, ori_mesh = ori_mesh)
 
    return out_dict

def output_model(net, input_tensor,ratio_h,ratio_w):
    
    batch_size, _, img_h, img_w = input_tensor.size()
    warp_h = int(img_h * ratio_h)
    warp_w = int(img_w * ratio_w)
    s_w = img_w/224
    s_h = img_h/224

    resized_input = resize_224(input_tensor)
    motion = net(resized_input,is_training=False)
   
   
    motion = torch.stack([motion[...,0]*s_w, motion[...,1]*s_h], 3)
   
    rigid_mesh = get_rigid_mesh(batch_size, warp_h, warp_w)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, warp_h, warp_w)
    
    mesh = rigid_mesh + motion
    norm_mesh = get_norm_mesh(mesh, warp_h, warp_w)
    
    stack_rigid_mesh = get_stack_mesh(norm_rigid_mesh)
    stack_mesh = get_stack_mesh(norm_mesh)
    
    out_dict = {}
    if (ratio_w <= 1):
        warp_pri = torch_tps_transform.transformer(input_tensor, stack_mesh,stack_rigid_mesh,  (int(warp_h), int(warp_w)))
    else:
        warp_pri = torch_tps_transform.transformer(input_tensor, stack_rigid_mesh, stack_mesh, (int(warp_h), int(warp_w)))

    out_dict.update(warp_primary = warp_pri, mesh_primary = mesh)
    return out_dict


# define and forward
class Network(nn.Module):

    def __init__(self,motion_init_value=None):
        super(Network, self).__init__()
        self.regressNet_part1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),        
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),        
            nn.MaxPool2d(2, 2),           

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),        
            nn.MaxPool2d(2, 2),                 
                   
            
        )
        
        self.regressNet_part2 = nn.Sequential(
            
            nn.Linear(in_features=1152,out_features=(grid_w+1)*(grid_h+1)*2, bias=True),#576
        )
       
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
       
        ssl._create_default_https_context = ssl._create_unverified_context
        
        resnet50_model = models.resnet.resnet50(pretrained=True)
        
        if torch.cuda.is_available():
            resnet50_model = resnet50_model.cuda()
        self.feature_extractor = self.get_res50_FeatureMap(resnet50_model)
    
    def get_res50_FeatureMap(self, resnet50_model):

        layers_list = []

        layers_list.append(resnet50_model.conv1)
        layers_list.append(resnet50_model.bn1)
        layers_list.append(resnet50_model.relu)
        layers_list.append(resnet50_model.maxpool)
        layers_list.append(resnet50_model.layer1)
        layers_list.append(resnet50_model.layer2)
        
        feature_extractor_stage = nn.Sequential(*layers_list)
        
        return feature_extractor_stage
    
        
    # forward
    def forward(self, input_tensor, is_training):
       
        b, _, img_h, img_w = input_tensor.size()
        input_tensor = (input_tensor + 1) / 2
        
        features = self.feature_extractor(input_tensor)
        features = features[:, -1, :, :].unsqueeze(1)
        
        temp1 = self.regressNet_part1(features)
        temp1 = temp1.reshape(temp1.size(0), -1)  
        mesh_shift_primary = self.regressNet_part2(temp1)   
        mesh_shift_primary = mesh_shift_primary.reshape(-1, grid_h+1, grid_w+1, 2)
        if is_training==False:    
            return mesh_shift_primary
        
        return mesh_shift_primary
     
        
   