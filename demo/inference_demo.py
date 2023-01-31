from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import numpy as np
import cv2
import glob
from tqdm import tqdm
from PIL import Image
from classification import * 
import time
##
# time 
start = time.time()

# config and check point file load
config_file = '/data2/JSLEE/Third_molar/demo/configs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes.py'
checkpoint_file = '/data2/JSLEE/Third_molar/demo/checkpoint/deeplabv3plus_r101-d8_512x1024_80k_cityscapes/iter_72000.pth'

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

######################################## img load ############################################
# test a single image
img = '/data2/JSLEE/Third_molar/demo/data/test/29430000.jpg.png'

# img_list = glob.glob("/data2/JSLEE/Third_molar/demo/data/test/*.png")
# for img in tqdm(img_list):
vis_img = cv2.imread(img)          
##############################################################################################
img_name = img.split("/")[-1].split(".")[0] + "." + img.split("/")[-1].split(".")[1]

result = inference_segmentor(model, img)

# list to numpy
result_np = np.array(result)
mask = result_np.transpose(1,2,0)

instances = np.unique(result_np)
instances = instances.tolist()


# #18 third molar
if 6 in instances:
    molar_18 = np.where(result_np == 6)
    y_min = int(np.min(molar_18[1]))
    y_max = int(np.max(molar_18[1]))
    x_min = int(np.min(molar_18[2]))
    x_max = int(np.max(molar_18[2]))
    
    if int(x_max-x_min) > 500:
        # data preprocess
        x_max = x_min+150
        x_mean = int((x_max + x_min)/2)
        y_mean = int((y_max+y_min)/2)
        crop_img = vis_img[y_mean-350:y_mean+350, x_mean-350:x_mean+350]
        mask_img = mask[y_mean-350:y_mean+350, x_mean-350:x_mean+350]
        # model 
        ex_di = maxil_ext(crop_img, mask_img)
        si_per = maxil_com(crop_img, mask_img)

    elif 50 <int(x_max-x_min) < 500:
        # data preprocess
        x_mean = int((x_max + x_min)/2)
        y_mean = int((y_max+y_min)/2)
        crop_img = vis_img[y_mean-350:y_mean+350, x_mean-350:x_mean+350]
        mask_img = mask[y_mean-350:y_mean+350, x_mean-350:x_mean+350]
        # model 
        ex_di = maxil_ext(crop_img, mask_img)
        si_per = maxil_com(crop_img, mask_img)

    elif int(x_max-x_min) < 50:
        pass
    
    print("#18 extraction difficulty: {} | possibilty of sinus perforation: {}".format(ex_di, si_per))

else:
    pass


if 8 in instances:
# 28 third molar
    molar_28 = np.where(result_np == 8)
    y_min = int(np.min(molar_28[1]))
    y_max = int(np.max(molar_28[1]))
    x_min = int(np.min(molar_28[2]))
    x_max = int(np.max(molar_28[2]))
  
    if int(x_max-x_min) > 500:
        #preprocess
        x_min = x_max-150
        x_mean = int((x_max+x_min)/2)
        y_mean = int((y_max+y_min)/2)
        crop_img = vis_img[y_mean-350:y_mean+350, x_mean-350:x_mean+350]
        mask_img = mask[y_mean-350:y_mean+350, x_mean-350:x_mean+350]
        #model
        ex_di = maxil_ext(crop_img, mask_img)
        si_per = maxil_com(crop_img, mask_img)
        
    elif 50 <int(x_max-x_min) < 500:
        #preprocess
        x_mean = int((x_max + x_min)/2)
        y_mean = int((y_max+y_min)/2)
        crop_img = vis_img[y_mean-350:y_mean+350, x_mean-350:x_mean+350]
        mask_img = mask[y_mean-350:y_mean+350, x_mean-350:x_mean+350]
        #model
        ex_di = maxil_ext(crop_img, mask_img)
        si_per = maxil_com(crop_img, mask_img)
        
    elif int(x_max-x_min) < 50:
        pass
    print("#28 extraction difficulty: {} | possibilty of sinus perforation: {}".format(ex_di, si_per))
    

else:
    pass

if 10 in instances:
# 38 third molar
    molar_38 = np.where(result_np == 10)
    y_min = int(np.min(molar_38[1]))
    y_max = int(np.max(molar_38[1]))
    x_min = int(np.min(molar_38[2]))
    x_max = int(np.max(molar_38[2]))
    
    if int(x_max-x_min) > 500:
        #preprocess
        x_min = x_max-150
        x_mean = int((x_max+x_min)/2)
        y_mean = int((y_max+y_min)/2)
        crop_img = vis_img[y_mean-350:y_mean+350, x_mean-350:x_mean+350]
        mask_img = mask[y_mean-350:y_mean+350, x_mean-350:x_mean+350]
        #model
        ex_di = mandi_ext(crop_img, mask_img)
        IAN_injury = mandi_com(crop_img, mask_img)
        
    elif 50 <int(x_max-x_min) < 500:
        #preprocess
        x_mean = int((x_max + x_min)/2)
        y_mean = int((y_max+y_min)/2)
        crop_img = vis_img[y_mean-350:y_mean+350, x_mean-350:x_mean+350]
        mask_img = mask[y_mean-350:y_mean+350, x_mean-350:x_mean+350]
        #model
        ex_di = mandi_ext(crop_img, mask_img)
        IAN_injury = mandi_com(crop_img, mask_img)
                
    elif int(x_max-x_min) < 50:
        pass

    print("#38 extraction difficulty: {} | possibilty of IAN Injury: {}".format(ex_di, IAN_injury))

else:
    pass
    
if 12 in instances:
# 48 third molar
    molar_48 = np.where(result_np == 12)
    y_min = int(np.min(molar_48[1]))
    y_max = int(np.max(molar_48[1]))
    x_min = int(np.min(molar_48[2]))
    x_max = int(np.max(molar_48[2]))

    if int(x_max-x_min) > 500:
        #preprocess
        x_max = x_min+150
        x_mean = int((x_max + x_min)/2)
        y_mean = int((y_max+y_min)/2)
        crop_img = vis_img[y_mean-350:y_mean+350, x_mean-350:x_mean+350]
        mask_img = mask[y_mean-350:y_mean+350, x_mean-350:x_mean+350]
        #model
        ex_di = mandi_ext(crop_img, mask_img)
        IAN_injury = mandi_com(crop_img, mask_img)
        
    elif 50 <int(x_max-x_min) < 500:
        #preprocess
        x_mean = int((x_max + x_min)/2)
        y_mean = int((y_max+y_min)/2)
        crop_img = vis_img[y_mean-350:y_mean+350, x_mean-350:x_mean+350]
        mask_img = mask[y_mean-350:y_mean+350, x_mean-350:x_mean+350]
        
        #model
        ex_di = mandi_ext(crop_img, mask_img)
        IAN_injury = mandi_com(crop_img, mask_img)
        
    elif int(x_max-x_min) < 50:
        pass
    
    print("#48 extraction difficulty: {} | possibilty of IAN Injury: {}".format(ex_di, IAN_injury))

else:
    pass

print("time spend:", time.time()- start)
