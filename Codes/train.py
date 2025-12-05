import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from network import build_model, Network
from dataset import TrainDataset
from loss import *
import glob
import os
from network import get_rigid_mesh,get_norm_mesh
from torch.autograd import gradcheck
import time
last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
# path to save the summary files
SUMMARY_DIR = os.path.join(last_path, 'summary')
writer = SummaryWriter(log_dir=SUMMARY_DIR)
# path to save the model files
MODEL_DIR = os.path.join(last_path, 'model')
# create folders if it dose not exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)


def train(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    batch_size=args.batch_size
    # define dataset
    train_data = TrainDataset(data_path=args.train_path)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=6, shuffle=True, drop_last=True)
    
    # define the network
    net = Network()
    if torch.cuda.is_available():
        net = net.cuda()
    
    # define the optimizer and learning rate
    optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)  # default as 0.0001
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    #load the existing models if it exists
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)

        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        glob_iter = checkpoint['glob_iter']
        scheduler.last_epoch = start_epoch
        print('load model from {}!'.format(model_path))
    else:
        start_epoch = 0
        glob_iter = 0
        print('training from stratch!')
    
    

    print("##################start training#######################")
    score_print_fre = 300
    
    
    start_time = time.time()
       
    for epoch in range(start_epoch, args.max_epoch):
        
        net.train()
        loss_sigma = 0.0
        content_loss_sigma = 0.
        grid_loss_sigma = 0.
        lab_loss_sigma = 0.
        bundary_loss_sigma = 0.

        print(epoch, 'lr={:.6f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        
        for i, batch_value in enumerate(train_loader):
            
            inpu_tensor = batch_value[0].float()

            if torch.cuda.is_available():
                inpu_tensor = inpu_tensor.cuda()

            # forward, backward, update weights
            optimizer.zero_grad()
            
  
            batch_out = build_model(net,inpu_tensor, is_training = True)
            warp_image = batch_out['warp_primary'] 
            rigid_mesh = batch_out['rigid_mesh']
            mesh = batch_out['mesh_pri']
            ori_mesh = batch_out['ori_mesh']           
            motion = batch_out['motion_primary']
           
            
            # calculate loss 
            content_loss, grid_loss = c_l(inpu_tensor, warp_image, ori_mesh, mesh)      
            bundary_loss = b_loss(warp_image, mesh, rigid_mesh, motion)
            total_loss = content_loss + grid_loss + bundary_loss
            total_loss.backward()
           
            # clip the gradient
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
            optimizer.step()
            
            
           
            content_loss_sigma += content_loss.item()
            grid_loss_sigma += grid_loss.item()
            bundary_loss_sigma += bundary_loss.item()
            loss_sigma += total_loss.item()
           
            print(glob_iter)
           
            
            # record loss and images in tensorboard
            if i % score_print_fre == 0 and i != 0:
                average_loss = loss_sigma / score_print_fre
                average_content_loss = content_loss_sigma/ score_print_fre
                average_grid_loss = grid_loss_sigma / score_print_fre
                average_bundary_loss = bundary_loss_sigma / score_print_fre
                     
                loss_sigma = 0.0
                content_loss_sigma = 0.
                grid_loss_sigma = 0.
                bundary_loss_sigma = 0.
                
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}]/[{:0>3}] Total Loss: {:.4f}  content Loss: {:.4f}  grid Loss: {:.4f}  bundary_loss:{:.4f}   lr={:.8f} ".format(epoch + 1, args.max_epoch, i + 1, len(train_loader),
                                          average_loss, average_content_loss, average_grid_loss, average_bundary_loss,  optimizer.state_dict()['param_groups'][0]['lr']))
                
                # visualization
               
                writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], glob_iter)
                writer.add_scalar('total loss', average_loss, glob_iter)
                writer.add_scalar('content loss', average_content_loss, glob_iter)
                writer.add_scalar('grid loss', average_grid_loss, glob_iter)
                writer.add_scalar('bundary loss', average_bundary_loss, glob_iter)
                
            
            glob_iter += 1
            
        
        scheduler.step()
        
        # save model
        
        if ((epoch+1) % 5 == 0 or (epoch+1)==args.max_epoch):   
            filename ='epoch' + str(epoch+1).zfill(3) + '_model.pth'
            model_save_path = os.path.join(MODEL_DIR, filename)
            state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1, "glob_iter": glob_iter}
            torch.save(state, model_save_path)
            
        
    end_time = time.time()
    inference_time = end_time - start_time    
    print("##################end training#######################", inference_time)


if __name__=="__main__":


    print('<==================== setting arguments ===================>\n')

    # create the argument parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--train_path', type=str, default='Object-IR/Data/train')
    # parse the arguments
    args = parser.parse_args()
    print(args)

    # train
    train(args)

