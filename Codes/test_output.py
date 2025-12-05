# coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
import imageio
from network import build_model,output_model, Network
from dataset import *
import os
import cv2
import predefine
import time
grid_h = predefine.GRID_H
grid_w = predefine.GRID_W

last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
MODEL_DIR = os.path.join(last_path, 'model')

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

def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(image_name)
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.5)
    return




def test(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # dataset
    test_data = OutDataset(data_path=args.test_path)
   
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=False)
    img_h, img_w = args.imgsize
    # define the network
    net = Network()#build_model(args.model_name)
    if torch.cuda.is_available():
        net = net.cuda()
    ratio_h = args.ratio_h
    ratio_w = args.ratio_w
    #load the existing models if it exists
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        
        checkpoint = torch.load(model_path, weights_only=True)

        net.load_state_dict(checkpoint['model'])
        print('load model from {}!'.format(model_path))
    else:
        print('No checkpoint found!')



    print("##################start testing#######################")
    # create folders if it dose not exist
    
    path_warp = '/mnt/wangr/Object-IR/warp/'
    if not os.path.exists(path_warp):
        os.makedirs(path_warp)
    path_grid = '/mnt/wangr/Object-IR/grid/'
    if not os.path.exists(path_grid):
        os.makedirs(path_grid)
   
    num_images = len(test_loader)
    total_time = 0.0
    net.eval()
    for i, batch_value in enumerate(test_loader):

        inpu_tensor = batch_value[0].float()
       
        if torch.cuda.is_available():
            inpu_tensor = inpu_tensor.cuda()          

        with torch.no_grad():
            start_time = time.time()

            batch_out = output_model(net,inpu_tensor,ratio_h,ratio_w)

            end_time = time.time()
            inference_time = end_time - start_time
            warp_image = batch_out['warp_primary']
            
            mesh = batch_out['mesh_primary']
        
       
   
   
        
        
       
        warp = ((warp_image[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
        
        path1 = path_warp + str(i+1).zfill(5) + ".jpg"
        cv2.imwrite(path1, warp)
        
        warp_draw = warp_image[0].cpu().detach().numpy().transpose(1,2,0)
        warp_draw = (warp_draw + 1.0) / 2.0 
        mesh = draw_mesh_on_warp(warp_draw, mesh[0].cpu().detach().numpy())
        pri_uint8 = (np.array(mesh) * 255)
 
        path2 = path_grid + str(i+1).zfill(5) + ".jpg"
        cv2.imwrite(path2, pri_uint8)


        

        print(f"Image {i+1} processed in {inference_time:.4f} seconds")
        
        total_time += inference_time
    
        torch.cuda.empty_cache()
   
    ave_time = total_time / num_images
    
    print("##################end testing#######################",f"average time is {ave_time:.4f} seconds")


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_path', type=str, default='Object-IR/Data/test')
    parser.add_argument('--ratio_h', type=float, default=1)
    parser.add_argument('--ratio_w', type=float, default=0.5)
    parser.add_argument('--imgsize', type=float, default=(224,224))
    
    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
    test(args)

