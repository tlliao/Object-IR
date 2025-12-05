import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import utils.torch_tps_transform as torch_tps_transform
from scipy.spatial.distance import cdist
import os
from torchvision.utils import save_image
import cv2
import numpy as np
import predefine
from ultralytics import YOLO
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
resize_224 = T.Resize((224,224))

grid_w = predefine.GRID_W
grid_h = predefine.GRID_H
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def c_l(input_tensor, warp_image, rigid_mesh, mesh): 
    
    def draw_boxes_on_images(image1, image2, batch_boxes1, batch_boxes2, color1=(0, 255, 0), color2=(0, 0, 255)):
       
        image1_np = (image1.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)  
        image2_np = (image2.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8) 
        image1_np = cv2.cvtColor(image1_np, cv2.COLOR_RGB2BGR)  
        image2_np = cv2.cvtColor(image2_np, cv2.COLOR_RGB2BGR)
       

        if image1_np.ndim == 2 or image1_np.shape[2] == 1:
            image1_np = cv2.cvtColor(image1_np, cv2.COLOR_GRAY2BGR)
        if image2_np.ndim == 2 or image2_np.shape[2] == 1:
            image2_np = cv2.cvtColor(image2_np, cv2.COLOR_GRAY2BGR)

        batch_boxes1 = batch_boxes1.int().cpu().detach().numpy()  
        batch_boxes2 = batch_boxes2.int().cpu().detach().numpy()  


        for box in batch_boxes1:
            x1, y1, x2, y2 = box
            cv2.rectangle(image1_np, (x1, y1), (x2, y2), color1, 2)  
        

        for box in batch_boxes2:
            x1, y1, x2, y2 = box
            cv2.rectangle(image2_np, (x1, y1), (x2, y2), color2, 2)  


        cv2.imwrite("map_boxes3.png", image1_np)
        cv2.imwrite("map_boxes4.png", image2_np)   
    def map_boxes(boxes, mesh, rigid_mesh, img_h, img_w, out_h, out_w):
        def find_grid_points(mesh, rigid_mesh, x, y, height, width, grid_h, grid_w):
            # Calculate step size
            
            step_h = height / grid_h
            step_w = width / grid_w

            # Find grid index
            a = int(y // step_h)
            b = int(x // step_w)

            # Clamp indices to stay within bounds
            a = max(0, min(a, grid_h - 1))
            b = max(0, min(b, grid_w - 1))

            # Extract four points of the grid
            p1 = rigid_mesh[i, a, b]       # Top-left
            p2 = rigid_mesh[i, a, b+1]     # Top-right
            p3 = rigid_mesh[i, a+1, b]     # Bottom-left
            p4 = rigid_mesh[i, a+1, b+1]   # Bottom-right
            p1_1 = mesh[i, a, b]       # Top-left
            p2_1 = mesh[i, a, b+1]     # Top-right
            p3_1 = mesh[i, a+1, b]     # Bottom-left
            p4_1 = mesh[i, a+1, b+1]   # Bottom-right
            # combine
           
            rigid_points_x = torch.stack([p1[0], p2[0], p3[0], p4[0]])
            rigid_points_y = torch.stack([p1[1], p2[1], p3[1], p4[1]])
            transformed_points_x = torch.stack([p1_1[0], p2_1[0], p3_1[0], p4_1[0]])
            transformed_points_y = torch.stack([p1_1[1], p2_1[1], p3_1[1], p4_1[1]])
            return rigid_points_x, transformed_points_x, rigid_points_y, transformed_points_y

        def transform_coefficients(x, y, x_prime, y_prime):
               
            A = torch.stack([torch.ones_like(x), x, y, x * y], dim=1)
            a, _ = torch.linalg.lstsq(A, x_prime.unsqueeze(-1))[:2]  
            b, _ = torch.linalg.lstsq(A, y_prime.unsqueeze(-1))[:2]  

            return a[:, 0], b[:, 0]

        def transform_coordinates(x, y, a, b):
         
            # Compute x' and y'
            x_prime = a[0] + a[1] * x + a[2] * y + a[3] * x * y
            y_prime = b[0] + b[1] * x + b[2] * y + b[3] * x * y

            return x_prime, y_prime
       
        num_boxes = boxes.size(0)
        
        map_boxes = []
        for n in range(num_boxes):
            x1, y1, x2, y2 = boxes[n]
            
            rigid_points_x1, transformed_points_x1, rigid_points_y1, transformed_points_y1 = find_grid_points(mesh, rigid_mesh, x1, y1, img_h, img_w, grid_h, grid_w)
            rigid_points_x2, transformed_points_x2, rigid_points_y2, transformed_points_y2 = find_grid_points(mesh, rigid_mesh, x2, y2, img_h, img_w, grid_h, grid_w)
            a1, b1 = transform_coefficients(rigid_points_x1, rigid_points_y1, transformed_points_x1, transformed_points_y1)
            a2, b2 = transform_coefficients(rigid_points_x2, rigid_points_y2, transformed_points_x2, transformed_points_y2)
            x1_t, y1_t = transform_coordinates(x1, y1, a1, b1)
            x2_t, y2_t = transform_coordinates(x2, y2, a2, b2)
            map_boxes.append([x1_t, y1_t, x2_t, y2_t])
            
        return torch.tensor(map_boxes, device=mesh.device, dtype=torch.float32) 
    def extract_region_with_padding(image1, box1, image2, box2, pad_value=0):
      
        _, img_h1, img_w1 = image1.size()
        _, img_h2, img_w2 = image2.size()

   
        x1_1, y1_1, x2_1, y2_1 = map(int, box1)
        x1_valid_1, y1_valid_1 = max(0, x1_1), max(0, y1_1)
        x2_valid_1, y2_valid_1 = min(img_w1, x2_1), min(img_h1, y2_1)

      
        x1_2, y1_2, x2_2, y2_2 = map(int, box2)
        x1_valid_2, y1_valid_2 = max(0, x1_2), max(0, y1_2)
        x2_valid_2, y2_valid_2 = min(img_w2, x2_2), min(img_h2, y2_2)

        region1_false = torch.zeros((image1.size(0), 0, 0)) 
        region2_false = torch.zeros((image2.size(0), 0, 0)) 
        
        region1 = image1[:, y1_valid_1:y2_valid_1, x1_valid_1:x2_valid_1]
        region2 = image2[:, y1_valid_2:y2_valid_2, x1_valid_2:x2_valid_2]
        
        
        if(region2.shape[1]==0 or region2.shape[2]==0):
            return region1_false, region2_false

        height1, width1 = region1.shape[1], region1.shape[2]
        height2, width2 = region2.shape[1], region2.shape[2]
        h_diff = height1 - height2
        w_diff = width1 - width2
        up_hw = min(h_diff, w_diff)
        if(up_hw == h_diff):
            h_s = height1 / height2
            region2 = F.interpolate(region2.unsqueeze(0), size=(region1.shape[1], max(1, int(region2.shape[2] * h_s))), mode='bilinear', align_corners=False)
            region2 = region2.squeeze(0)
        else:
            w_s = width1 / width2
            region2 = F.interpolate(region2.unsqueeze(0), size=(max(1, int(region2.shape[1] * w_s)), region1.shape[2]), mode='bilinear', align_corners=False)
            region2 = region2.squeeze(0)
  
        max_width = max(width1, width2)
        max_height = max(height1, height2)
        
        if width1 < max_width:
           
            padding_left = max(0, (max_width - width1) // 2)
            padding_right = max(0, max_width - width1 - padding_left)

            if padding_left > x1_valid_1:
                
                padding_left = x1_valid_1 
                padding_right = min((max_width - width1 - padding_left),(img_w1 - width1 - padding_left))
 
            if x1_valid_1 + width1 + padding_right > img_w1:
                
                padding_right = img_w1 - (x1_valid_1 + width1)
                padding_left = min((max_width - width1 - padding_right),(img_w1 - width1 - padding_right))
        
            region1 = F.pad(region1, (padding_left, padding_right))  

        if width2 < max_width:
            padding_left = max(0, (max_width - width2) // 2)
            padding_right = max(0, max_width - width2 - padding_left)
           
            if padding_left > x1_valid_2:
               
                padding_left = x1_valid_2  
                padding_right = min((max_width - width2 - padding_left),(img_w2- width2-padding_left))  

            if x1_valid_2 + width2 + padding_right > img_w2:
             
                padding_right = img_w2 - (x1_valid_2 + width2)
                padding_left = min((max_width - width2 - padding_right),(img_w2 - width2 - padding_right)) 
            
            region2 = F.pad(region2, (padding_left, padding_right))  

        if height1 < max_height:
           
            padding_top = max(0, (max_height - height1) // 2)
            padding_bottom = max(0, max_height - height1 - padding_top)
               
            if padding_top > y1_valid_1:
              
                padding_top = y1_valid_1  
                padding_bottom = min((max_height - height1 - padding_top),(img_h1 - height1 - padding_top))
       
            if y1_valid_1 + height1 + padding_bottom > img_h1:
               
                padding_bottom = img_h1 - (y1_valid_1 + height1)  
                padding_top =  min((max_height - height1 - padding_bottom),(img_h1 - height1 - padding_bottom))

            region1 = F.pad(region1, (0, 0, padding_top, padding_bottom))  
       
        if height2 < max_height:
            padding_top = max(0, (max_height - height2) // 2)
            padding_bottom = max(0, max_height - height2 - padding_top)
           
            if padding_top > y1_valid_2:
               
                padding_top = y1_valid_2  
                padding_bottom = min((max_height - height2 - padding_top),(img_h2 - height2 - padding_top))

            if y1_valid_2 + height2 + padding_bottom > img_h2:
               
                padding_bottom = img_h2 - (y1_valid_2 + height2)
                padding_top =  min((max_height - height2 - padding_bottom),(img_h2 - height2 - padding_bottom)) 
            
            region2 = F.pad(region2, (0, 0, padding_top, padding_bottom)) 
        
        if region1.size() != region2.size():               
            f_size = (region2.shape[1], region2.shape[2])
            region1 = F.interpolate(region1.unsqueeze(0), size=f_size, mode='bilinear', align_corners=False)
            region1 = region1.squeeze(0)
        return region1, region2
    
    def save_image_with_padding(region, filename):
        """
        save region as an image
        """
        # transform PyTorch tensor to NumPy array
        region_np = region.permute(1, 2, 0).cpu().detach().numpy()  # 转换为 HWC 格式

        # restrict the pixel value into [0, 255]
        region_np = (region_np * 255).astype(np.uint8)  # suppose pixel value lies in [0, 1]

        # change gray image to 3 channel
        if region_np.shape[2] == 1:
            region_np = np.repeat(region_np, 3, axis=2)

        # make save path
        #output_path = os.path.join(output_folder, filename)
        
        # save image
        cv2.imwrite(filename, region_np)
    def compute_boxwise(input_tensor, warp_image, boxes1, boxes2):
        _, w_h, w_w = warp_image.size()
        count1 = 0
        num_boxes = boxes1.size(0)
        all_p_loss = 0.0

        all_xy_loss = 0.0
        for n in range(num_boxes):
           
            region1, region2 = extract_region_with_padding(input_tensor, boxes1[n], warp_image, boxes2[n], pad_value=0)
    
            if region1.size(1) == 0 or region1.size(2) == 0 or region2.size(1) == 0 or region2.size(2) == 0:
                #print("------zero region------")
                continue
          
            region1 = resize_224(region1)
            region2 = resize_224(region2)
            #save_image_with_padding(region1, "region1.png")
            #save_image_with_padding(region2, "region2.png")
            
            region1 = region1.to(device)
            region2 = region2.to(device)   
            mse_loss = F.mse_loss(region1, region2)    
            contentl = mse_loss*2000
            all_p_loss += contentl
            
            ############GRID LOSS#################
            x1, y1, x2, y2 = boxes1[n]
            
            x1_grid = (x1 * grid_w / img_w).int()
            y1_grid = (y1 * grid_h / img_h).int()
            x2_grid = (x2 * grid_w / img_w).int()
            y2_grid = (y2 * grid_h / img_h).int()
            x1_grid = max(0, min(x1_grid, grid_w))
            y1_grid = max(0, min(y1_grid, grid_h))
            x2_grid = max(0, min(x2_grid, grid_w))
            y2_grid = max(0, min(y2_grid, grid_h))
            if(x1_grid == x2_grid != grid_w or y1_grid == y2_grid != grid_h):
                x2_grid = x1_grid + 1
            elif(x1_grid == x2_grid == grid_w or y1_grid == y2_grid == grid_h):
                x1_grid = x2_grid - 1
            
            mesh_b = mesh[i, y1_grid:y2_grid+1, x1_grid:x2_grid+1, :]
            rigid_b = rigid_mesh[i, y1_grid:y2_grid+1, x1_grid:x2_grid+1, :]
            
            if rigid_b.size(1) > 1:  # make sure there are at least two elements
                delta_x_rigid = torch.diff(rigid_b[..., 0], dim=1)
                delta_y_rigid = torch.diff(rigid_b[..., 1], dim=1)
            else:
                
                continue

            if mesh_b.size(1) > 1:  # make sure there are at least two elements
                delta_x_model = torch.diff(mesh_b[..., 0], dim=1)
                delta_y_model = torch.diff(mesh_b[..., 1], dim=1)
            else:
                
                continue
            
            x_loss = torch.mean(torch.abs(para_w * delta_x_rigid - delta_x_model))
            y_loss = torch.mean(torch.abs(para_h * delta_y_rigid - delta_y_model))
            xy_loss = x_loss + y_loss
                    
            all_xy_loss += xy_loss              
            count1 += 1
        if(count1 != 0) :   
            all_p_loss /= count1 
            all_xy_loss /= count1    
            return all_p_loss, all_xy_loss 
        else:
            return torch.tensor([]), torch.tensor([])
        
    img_h, img_w = input_tensor.shape[2:]
    out_h, out_w = warp_image.shape[2:]
    batch_size = input_tensor.size(0)
    r_w = out_w / img_w
    r_h = out_h / img_h
    h_in_256 = 256
    w_in_256 = 256
    input_256 = F.interpolate(input_tensor, size=(h_in_256, w_in_256), mode='bilinear', align_corners=False)
    input_256 = (input_256 + 1) / 2
    input_tensor = (input_tensor + 1) / 2  
    warp_image = (warp_image + 1) / 2 
    
    model_yolo = YOLO('/mnt/wangr/Object-IR/Codes/yolo11n.pt')
    results_img = model_yolo.predict(input_256,verbose=False)

    para_w = torch.sqrt(torch.tensor(r_w))   
    para_h = 1
    

    count2 = 0
    p_loss = 0.0 
    g_loss = 0.0
    for i in range(batch_size): 
        
        batch_boxes = results_img[i].boxes.xyxy        
       
        if batch_boxes.shape[0] == 0 :
            continue

        def calculate_area(boxes):
           
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            width = x2 - x1
            height = y2 - y1
            area = width * height
            return area

        
        areas = calculate_area(batch_boxes)
        min_area = int((h_in_256 * w_in_256) / 10)
        max_area = int((h_in_256 * w_in_256) / 2)
    
        batch_boxes = batch_boxes[ (max_area > areas) & (areas > min_area)]    
        if batch_boxes.shape[0] == 0 :
            continue
        scale_x_in = img_w / w_in_256
        scale_y_in = img_h / h_in_256
    

        batch_boxes[:, [0, 2]] = batch_boxes[:, [0, 2]] * scale_x_in  
        batch_boxes[:, [1, 3]] = batch_boxes[:, [1, 3]] * scale_y_in  
       

        mapped_boxes = map_boxes(batch_boxes, mesh, rigid_mesh, img_h, img_w, out_h, out_w)
        draw_boxes_on_images(input_tensor[i], warp_image[i], batch_boxes, mapped_boxes, color1=(0, 255, 0), color2=(0, 0, 255))    
        if mapped_boxes.shape[0] == 0 :
            continue    

        loss_batch_c, loss_batch_g = compute_boxwise(input_tensor[i], warp_image[i], batch_boxes, mapped_boxes)
        if( loss_batch_c.numel() == 0 or loss_batch_g.numel() == 0):
            continue
        p_loss += loss_batch_c
        g_loss += loss_batch_g
        count2 += 1
    if count2 > 0:
        p_loss /= (count2 * 2)
        g_loss /= (count2)
    
    if not isinstance(p_loss, torch.Tensor):
        print("--------no tensor--------")
        p_loss = torch.tensor(p_loss, dtype=torch.float32, requires_grad=True, device=input_tensor.device)
    if not isinstance(g_loss, torch.Tensor):
        print("--------no tensor--------")
        g_loss = torch.tensor(g_loss, dtype=torch.float32, requires_grad=True, device=input_tensor.device)
    if torch.isnan(p_loss).any():
        print("-----nan-----")
        return torch.tensor(0.0, requires_grad=True, device=input_tensor.device)
    if torch.isnan(g_loss).any():
        print("-----nan-----")
        return torch.tensor(0.0, requires_grad=True, device=input_tensor.device)

    return p_loss, g_loss



def b_loss(warp_image, mesh, rigid_mesh, motion):
    
    out_h,out_w = warp_image.shape[2:]
    
    loss_tb_y = (motion[:, 0, :, 1].abs().mean() + motion[:, -1, :, 1].abs().mean()) / 2
    rigid_x_length = out_w / grid_w
    motion_x_top = motion[:, 0, 1:, 0].abs()
    motion_x_bottom = motion[:, -1, 1:, 0].abs()
    loss_tb_x = torch.relu(motion_x_top - 0.5 * rigid_x_length).mean() + \
                torch.relu(motion_x_bottom - 0.5 * rigid_x_length).mean()
    
    loss_lr_x = (motion[:, :, 0, 0].abs().mean() + motion[:, :, -1, 0].abs().mean()) / 2
    rigid_y_length = out_h / grid_h
    motion_y_left = motion[:, :-1, 0, 1].abs()
    motion_y_right = motion[:, :-1, -1, 1].abs()
    loss_lr_y = torch.relu(motion_y_left - 0.5 * rigid_y_length).mean() + \
                torch.relu(motion_y_right - 0.5 * rigid_y_length).mean()


    b_loss = loss_tb_y + loss_tb_x + loss_lr_x + loss_lr_y 
  
    return b_loss

