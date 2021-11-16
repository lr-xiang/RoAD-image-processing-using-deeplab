import argparse
import os
import numpy as np
import time

from dataloaders import custom_transforms as tr
from PIL import Image

from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
from skimage import morphology 

import pandas as pd

from skimage.color import rgb2hsv

# usage
# python ./measure.py /media/lietang/easystore1/RoAD/exp20 

# python ./measure.py /media/lietang/easystore1/RoAD/exp20 --test 1 --test_label S345-3_W_55.91_3 --test_date 2021-8-30


# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')

# Optional positional argument
parser.add_argument('exp_path', type=str,
                    help='An required integer positional argument')
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
print(args.exp_path)
print(args.test)
print(args.test_label)   
print(args.test_date)  


exp_path = args.exp_path
date_file = os.path.join(exp_path, "experiment_date.txt")
dates = open(date_file).read().splitlines()

label_file = os.path.join(exp_path, "experiment_label.txt")
labels = open(label_file).read().splitlines()

csv_path = os.path.join(exp_path, '2d_image_deeplab.csv')

# camera parameters
pixelSize=5.86e-04; #cm
fx=1.4135021364891049e+03; #pixel
fc=15.24; #cm; =6 inch
scaleFactor= fc/(fx*pixelSize)*pixelSize; #cm/pixel

traits = ['label', 'date', 
          'area_2d','convex_area_2d','bbox_area_2d',
          'perimeter_2d',
          'major_axis_length_2d',
          'minor_axis_length_2d',
          'solidity_2d',
          'eccentricity_2d',
          ]
traits = np.append(traits, 'abs_area_2d')  

healthy_traits = ['R_healthy', 'G_healthy', 'B_healthy', 'H_healthy', 'S_healthy', 'V_healthy', 'area_healthy_2d']
wilted_traits = ['R_wilted', 'G_wilted', 'B_wilted', 'H_wilted', 'S_wilted', 'V_wilted', 'area_wilted_2d']
dry_traits = ['R_dry', 'G_dry', 'B_dry', 'H_dry', 'S_dry', 'V_dry', 'area_dry_2d']

traits = np.append(traits, healthy_traits)
traits = np.append(traits, wilted_traits)
traits = np.append(traits, dry_traits)

# traits = np.append(traits, 'abs_day')  


df = pd.DataFrame( columns = traits)

for label in labels:
    if args.test == 1:
        if label != args.test_label:
            continue
            
    print(label)
    directory = os.path.join(exp_path, '2d_images_deeplab', label, 'processed')
    day_count = 0
    for date in dates:
        if args.test == 1:
            if date!= args.test_date:
                continue
        print(date)
        whole_plant = np.zeros([1200, 1920], int)
        
        green_path = os.path.join(directory, label+'_'+date+'_plant.png')
        purple_path = os.path.join(directory, label+'_'+date+'_wilted.png')
        yellow_path = os.path.join(directory, label+'_'+date+'_dry.png')

#         props of the whole plant
        if os.path.exists(green_path):
            green = Image.open(green_path)
            green = np.array(green)
            whole_plant[np.where(green[:,:,0]>0)] = 1
            
        if os.path.exists(purple_path):
            purple = Image.open(purple_path)
            purple = np.array(purple)
            whole_plant[np.where(purple[:,:,0]>0)] = 1
            
        if os.path.exists(yellow_path):
            yellow = Image.open(yellow_path)
            yellow = np.array(yellow)
            whole_plant[np.where(yellow[:,:,0]>0)] = 1
            
        if(np.count_nonzero(whole_plant)<10):
            continue
    
        mask = whole_plant>0
#         print(np.count_nonzero(mask))
        whole_plant = morphology.remove_small_objects(mask, 200).astype(np.uint8)
#         remove small components 
#         plt.imshow(whole_plant)
    
        props = regionprops_table(whole_plant, properties=('area',
                                            'convex_area',
                                            'bbox_area',
                                            'perimeter',
                                            'major_axis_length',
                                            'minor_axis_length',
                                            'solidity',
                                            'eccentricity',                                                        
                                            )
                         )
        df_tmp = pd.DataFrame(props)
        
        df_tmp['label'] = label
        df_tmp['date'] = date
        
#         convert to actual length
        df_tmp['area'] = df_tmp['area']*scaleFactor*scaleFactor
        df_tmp['convex_area'] = df_tmp['convex_area']*scaleFactor*scaleFactor
        df_tmp['bbox_area'] = df_tmp['bbox_area']*scaleFactor*scaleFactor
        df_tmp['perimeter'] = df_tmp['perimeter']*scaleFactor
        df_tmp['major_axis_length'] = df_tmp['major_axis_length']*scaleFactor
        df_tmp['minor_axis_length'] = df_tmp['minor_axis_length']*scaleFactor
    
        df_tmp.rename(columns = {'area':'area_2d', 'convex_area':'convex_area_2d',
                             'bbox_area':'bbox_area_2d', 'perimeter':'perimeter_2d',
                             'major_axis_length':'major_axis_length_2d', 'minor_axis_length':'minor_axis_length_2d',
                             'solidity':'solidity_2d', 'eccentricity':'eccentricity_2d'}
                             ,inplace = True)
        if day_count == 0:
            df_tmp['abs_area_2d'] = 0
        else:
            df_tmp['abs_area_2d'] = df_tmp['area_2d'] - last_area
            
        last_area = df_tmp['area_2d']
        
        if os.path.exists(green_path):
            green = Image.open(green_path)
            arr = np.array(green)
            
            arr_r = arr[:,:,0]
            arr_g = arr[:,:,1]
            arr_b = arr[:,:,2]

            df_tmp['R_healthy'] = arr_r[np.nonzero(arr_r)].mean() 
            df_tmp['G_healthy'] = arr_g[np.nonzero(arr_g)].mean() 
            df_tmp['B_healthy'] = arr_b[np.nonzero(arr_b)].mean() 
            
            hsv = rgb2hsv(arr)

            arr_h = hsv[:,:,0]
            arr_s = hsv[:,:,1]
            arr_v = hsv[:,:,2]
            
            df_tmp['H_healthy'] = arr_h[np.nonzero(arr_h)].mean() 
            df_tmp['S_healthy'] = arr_s[np.nonzero(arr_s)].mean() 
            df_tmp['V_healthy'] = arr_v[np.nonzero(arr_v)].mean() 
            
            df_tmp['area_healthy_2d'] = np.count_nonzero(arr_g)*scaleFactor*scaleFactor
        else:
            df_tmp[healthy_traits] = 0        
            
        if os.path.exists(purple_path):
            purple = Image.open(purple_path)
            arr = np.array(purple)
            
            arr_r = arr[:,:,0]
            arr_g = arr[:,:,1]
            arr_b = arr[:,:,2]

            df_tmp['R_wilted'] = arr_r[np.nonzero(arr_r)].mean() 
            df_tmp['G_wilted'] = arr_g[np.nonzero(arr_g)].mean() 
            df_tmp['B_wilted'] = arr_b[np.nonzero(arr_b)].mean() 
            
            hsv = rgb2hsv(arr)

            arr_h = hsv[:,:,0]
            arr_s = hsv[:,:,1]
            arr_v = hsv[:,:,2]
            
            df_tmp['H_wilted'] = arr_h[np.nonzero(arr_h)].mean() 
            df_tmp['S_wilted'] = arr_s[np.nonzero(arr_s)].mean() 
            df_tmp['V_wilted'] = arr_v[np.nonzero(arr_v)].mean() 
            
            df_tmp['area_wilted_2d'] = np.count_nonzero(arr_g)*scaleFactor*scaleFactor
        else:
            df_tmp[wilted_traits] = 0
            
        if os.path.exists(yellow_path):
            yellow = Image.open(yellow_path)
            arr = np.array(yellow)
            
            arr_r = arr[:,:,0]
            arr_g = arr[:,:,1]
            arr_b = arr[:,:,2]

            df_tmp['R_dry'] = arr_r[np.nonzero(arr_r)].mean() 
            df_tmp['G_dry'] = arr_g[np.nonzero(arr_g)].mean() 
            df_tmp['B_dry'] = arr_b[np.nonzero(arr_b)].mean() 
            
            hsv = rgb2hsv(arr)

            arr_h = hsv[:,:,0]
            arr_s = hsv[:,:,1]
            arr_v = hsv[:,:,2]
            
            df_tmp['H_dry'] = arr_h[np.nonzero(arr_h)].mean() 
            df_tmp['S_dry'] = arr_s[np.nonzero(arr_s)].mean() 
            df_tmp['V_dry'] = arr_v[np.nonzero(arr_v)].mean() 
            
            df_tmp['area_dry_2d'] = np.count_nonzero(arr_g)*scaleFactor*scaleFactor
        else:
            df_tmp[dry_traits] = 0
            
        df = df.append(df_tmp)
        
        day_count = day_count + 1

print(df)
if args.test != 1:
	df.to_csv(csv_path, sep='\t', index = False)
