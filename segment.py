import argparse
import os
import numpy as np
import time

from modeling.deeplab import *
from dataloaders import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from dataloaders.utils import  *
from torchvision.utils import make_grid, save_image
import matplotlib.gridspec as gridspec
import argparse

# usage
# python ./segment.py run/arab3/deeplab-drn/model_best_6816.pth.tar /media/lietang/easystore1/RoAD/exp20 

# python ./segment.py run/arab3/deeplab-drn/model_best_6816.pth.tar /media/lietang/easystore1/RoAD/exp20 --test 1 --test_label S345-3_W_55.91_3 --test_date 2021-8-30

# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')

# Required positional argument
parser.add_argument('model_path', type=str,
                    help='A required trained weights')

# Optional positional argument
parser.add_argument('exp_path', type=str, nargs='?',
                    help='An optional integer positional argument')

# Optional argument
parser.add_argument('--test', type=int,
                    help='An optional integer argument')            
    
# Optional argument
parser.add_argument('--test_label', type=str,
                    help='An optional integer argument')
                                    
# Optional argument
parser.add_argument('--test_date', type=str,
                    help='An optional integer argument')      
                    
args = parser.parse_args()

print("Argument values:")
print(args.model_path)
print(args.exp_path)
print(args.test)
print(args.test_label)   
print(args.test_date)  

# create model
model_s_time = time.time()
model = DeepLab(num_classes=4,
                backbone='drn',
                output_stride=16,
                sync_bn=None,
                freeze_bn=False)
                
model_path = args.model_path #"run/arab3/deeplab-drn/model_best_6816.pth.tar" 
ckpt = torch.load(model_path, map_location='cpu')
model.load_state_dict(ckpt['state_dict'])
model = model.cuda()
model_u_time = time.time()
model_load_time = model_u_time-model_s_time
print("model load time is {}".format(model_load_time))


#seg for experiment
exp_path = args.exp_path #"/media/lietang/easystore1/RoAD/exp20"
date_file = os.path.join(exp_path, "experiment_date.txt")
dates = open(date_file).read().splitlines()

label_file = os.path.join(exp_path, "experiment_label.txt")
labels = open(label_file).read().splitlines()

fig = plt.figure(figsize=(16, 8), dpi=80)

        
pix_th = 200
# MASK, 0-bg, 38-green, 75-purple, 113-yellow
composed_transforms = transforms.Compose([
    tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    tr.ToTensor()])

for label in labels:
    if args.test == 1:
        if label != args.test_label:
            continue
                      
    for date in dates:
        if args.test == 1:
            if date!= args.test_date:
                continue
        img_path = os.path.join(exp_path, date, label, label+".bmp")
        if not os.path.exists(img_path):
            continue
            
        s_time = time.time()
        image = Image.open(img_path).convert('RGB')
        target = Image.open(img_path).convert('L')
        sample = {'image': image, 'label': target}
        tensor_in = composed_transforms(sample)['image'].unsqueeze(0)

        model.eval()
        tensor_in = tensor_in.cuda()
        with torch.no_grad():
            output = model(tensor_in)
            
        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy()),
                                3, normalize=False, range=(0, 255))

        u_time = time.time()
        img_time = u_time-s_time
        print("image:{} time: {} ".format(img_path,img_time))

        save_image(grid_image,"mask2.png")
        mask = Image.open("mask2.png").convert('L')
        mask = np.array(mask)
        print('unique: ', np.unique(mask))
        plt.subplot(2, 3, 1)
        plt.imshow(mask)
        
        img = np.array(image)
        img_purple = np.array(image)
        img_yellow= np.array(image)
#         img[np.where(mask==0)] = [0, 0, 0] #255, 255, 255
        img[np.where(mask!=38)] = [0, 0, 0] 
        img_purple[np.where(mask!=75)] = [0, 0, 0] 
        img_yellow[np.where(mask!=113)] = [0, 0, 0] 
#    
# added on 11122021
        mask[:, :400] = 0
        mask[:, 1600:] = 0
        
        img[:, :400] = 0
        img[:, 1600:] = 0
        img_purple[:, :400] = 0
        img_purple[:, 1600:] = 0
        img_yellow[:, :400] = 0
        img_yellow[:, 1600:] = 0

        import os
        directory = os.path.join(exp_path, '2d_images_deeplab', label, 'processed')
#         print(directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
    
        plt.subplot(2, 3, 2)
        plt.imshow(image)
            
        if (np.count_nonzero(mask==38))>pix_th:
            outpath = os.path.join(directory, label+'_'+date+'_plant.png')
            img = Image.fromarray(img)
            img.save(outpath)
            plt.subplot(2, 3, 3)
            
            plt.gca().set_title('green')
            plt.imshow(img)
            print('plant: ', np.count_nonzero(mask==38))

            
        if (np.count_nonzero(mask==75))>pix_th:
            outpath = os.path.join(directory, label+'_'+date+'_wilted.png')
            img = Image.fromarray(img_purple)
            img.save(outpath)
            plt.subplot(2, 3, 4)
            plt.gca().set_title('purple')
            plt.imshow(img)
            
        if (np.count_nonzero(mask==113))>pix_th:
            outpath = os.path.join(directory, label+'_'+date+'_dry.png')
            img = Image.fromarray(img_yellow)
            img.save(outpath)
            plt.subplot(2, 3, 5)
            plt.gca().set_title('yellow')
            plt.imshow(img)
            
        print(img_path)
        if args.test == 1:
        	plt.show()
#         break
#     break
