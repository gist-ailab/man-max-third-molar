from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
import torchvision
from torch.autograd import Variable
from torch.autograd import Function
import torch.backends.cudnn as cudnn
from PIL import Image
import loader_up
import glob
import cv2
import torch.optim as optim
from tqdm import tqdm
import time
import torchvision.models as models
import albumentations
from albumentations.pytorch.transforms import ToTensor
import json
import random
import timm
from imbalanced_dataset_sampler.torchsampler.imbalanced import ImbalancedDatasetSampler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

matplotlib.use('Agg')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
print(time.strftime('%Y-%m-%d', time.localtime(time.time())))
# Learning
parser.add_argument("--learning_rate", default=3e-2, type=float,
                    help="The initial learning rate for SGD.")
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epoch_num', type=int, default=300)
parser.add_argument('--gpu_id', type=str ,default='4')
parser.add_argument('--model_name', type=str, default='resnet')
parser.add_argument("--model", choices=["Vision_Transformer", "resnet34", "resnet152d","seresnet101", "efficientnet_b3", "R50-ViT-B_16","coat","efficientnet_l2"],
                    default="Vision_Transformer",help="Which model to use.")
parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16","ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                    default="ViT-B_16",help="Which variant to use.")
parser.add_argument("--img_size", default=384, type=int, help="Resolution size")
parser.add_argument("--pretrained_dir", type=str, default="Vit_pretrain/imagenet21k/imagenet21k_ViT-B_16.npz", 
                    help="Where to search for pretrained ViT models.")
parser.add_argument("--weight_decay", default=0, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")
parser.add_argument('--he_type', type=str, default='CLAHE', choices=['CLAHE', 'HE'])
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam','SGD','AdamW'])
#CBAM
parser.add_argument('--depth', type=int, default=50, choices=[18,34,50,101])
parser.add_argument('--att_type', type=str, default='CBAM', choices=['CBAM','BAM','basic'])
args = parser.parse_args()

save_dir = "./weight/maxil_com/PNG+mask_final_modi/checkpoint_model_{}_lr_{}_batchsize_{}".format(args.model_name, args.learning_rate, args.batch_size)
#save_dir = "./weight/maxil_ext/PNG_segmask/checkpoint_model_{}_lr_{}_batchsize_{}".format(args.model_name, args.learning_rate, args.batch_size)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# set seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

transform = albumentations.Compose([
                albumentations.Resize(args.img_size,args.img_size),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.Rotate(limit=(-30,30), p=0.5),
                #albumentations.ColorJitter(brightness=0.5, contrast=0.5, p=0.5),
                #albumentations.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                #albumentations.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=1.0),
                #albumentations.Cutout(num_holes=16, max_h_size=48, max_w_size=48, p=0.5),
                albumentations.ShiftScaleRotate(shift_limit=(-0.15, 0.15), scale_limit=1.0, rotate_limit=0, p=0.5)
                #albumentations.CLAHE(clip_limit=(4.0,4.0), tile_grid_size=(8,8), p=1.0)
                ])

transform2 = albumentations.Compose([
                albumentations.Resize(args.img_size,args.img_size),
                #albumentations.CLAHE(clip_limit=(4.0,4.0), tile_grid_size=(8,8), p=1.0)
                ])

train_dir = []
test_dir = []
train_imgs = glob.glob("./dataset/crop_PNG_Images/train_max/*.png")
val_imgs = glob.glob("./dataset/crop_PNG_Images/val_max/*.png")
test_imgs = glob.glob("./dataset/crop_PNG_Images/test_max/*.png")

train_dir = train_imgs + val_imgs
test_dir = test_imgs

train_dataset = loader_up.UpDataLoader_com(train_dir, set="train", he=args.he_type, transform=transform)
test_dataset = loader_up.UpDataLoader_com(test_dir, set="test", he=args.he_type, transform=transform2)
train_loader = torch.utils.data.DataLoader(train_dataset, sampler=ImbalancedDatasetSampler(train_dataset), batch_size = args.batch_size ,shuffle=False, num_workers=4)
#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size ,shuffle=False, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle= False, num_workers=1)        

num_classes=3

# model choose
# Vision Transformer 
if args.model == 'Vision_Transformer':
    #model = timm.models.vision_transformer.vit_base_patch16_224_in21k(pretrained=True, num_classes=4)
    #model = timm.models.vision_transformer_hybrid.vit_base_r50_s16_224_in21k(pretrained=True, num_classes=4)
    #model = timm.models.vision_transformer_hybrid.vit_large_r50_s32_224_in21k(pretrained=True, num_classes=4)
    model = timm.models.vision_transformer_hybrid.vit_large_r50_s32_384(pretrained=True, num_classes=3)
# elif args.model == 'R50+ViT_384'
#     from Vit.modeling import VisionTransformer, CONFIGS
#     config = CONFIGS[args.model_type]
#     model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
#     model.load_from(np.load(args.pretrained_dir))

# elif args.model == 'R50+ViT_224'
#     from Vit.modeling import VisionTransformer, CONFIGS
#     config = CONFIGS[args.model_type]
#     model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
#     model.load_from(np.load(args.pretrained_dir))

elif args.model == 'resnet34':
    model = timm.models.resnet.resnet34(pretrained=True, num_classes=3)
elif args.model == 'resnet152d':
    model = timm.models.resnet.resnet152d(pretrained=True, num_classes=3)
elif args.model == 'seresnet101':
    model = timm.models.resnet.seresnet101(pretrained=True, num_classes=3)
elif args.model == 'efficientnet_l2':
    model = timm.models.efficientnet.efficientnet_l2(pretrained=True, num_classes=3)
elif args.model == 'coat':
    model = timm.models.coat.coat_lite_small(pretrained=True, num_classes=3)

model.cuda()

criterion = nn.CrossEntropyLoss()

# Prepare optimizer and scheduler
if args.optimizer == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

elif args.optimizer == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)  #Vit

elif args.optimizer == "AdamW":
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)   #Deit

val = []
acc = []
epoch_list = []
y_true = []
y_pred = []
y_score = []
target_name = ['very easy', 'easy', 'hard']

for epoch in range(args.epoch_num):
    train_total = 0
    train_correct = 0
    confusion_matrix_train = torch.zeros(3, 3)
    train_bar = tqdm(train_loader)
    for train_iter, (images, label) in enumerate(train_bar):

        model.train()
        images = Variable(images).cuda()
        label = Variable(label).cuda()
     
        #Forward pass
        output, prob = model(images)
   
        ce_loss = criterion(output, label)

        loss = ce_loss

        _, predicted = torch.max(output.data, 1)
    
        train_total += label.size(0)
        train_correct += (predicted == label).sum().item()

        # Backward and optimize
        optimizer.zero_grad()
    
        loss.backward()
      
        optimizer.step()
        
        train_acc = 100 * train_correct / train_total
        for t, p in zip(label.view(-1), predicted.view(-1)):
            confusion_matrix_train[t.long(), p.long()] += 1

        train_bar.set_description(desc='[%d/%d] Loss: %.4f |Accuracy: %.2f' %
                                       (epoch, args.epoch_num - 1, loss.item(), train_acc))

        y_true = []
        y_pred = []
        if (train_iter + 1) % (len(train_loader) - 1) == 0:

            model.eval()

            with torch.no_grad():
                test_total = 0
                test_correct = 0
                test_bar = tqdm(test_loader)
                confusion_matrix = torch.zeros(3, 3)
                for test_iter, (images, label) in enumerate(test_bar):
                    images = Variable(images).cuda()
                    label = Variable(label).cuda()
                
                    output, prob = model(images)

                    _, predicted = torch.max(output.data, 1)
            
                    test_loss = criterion(output, label)
                    test_total += label.size(0)
                    test_correct += (predicted == label).sum().item()

                    y_true.append(label.item())
                    y_pred.append(predicted.item())
         

                    y_score.append(prob[0].cpu().numpy())
            
                    for t, p in zip(label.view(-1), predicted.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1

                    test_acc = 100 * test_correct / test_total
   
                    test_bar.set_description(desc='[%d/%d]  Loss: %.4f Accuracy: %.2f' %
                                                    (epoch, args.epoch_num - 1, test_loss.item(), test_acc))

                val.append(test_loss.item())
                acc.append(test_acc)
        
                epoch_list.append(epoch)

                per_class_acc = np.round((100 * confusion_matrix.diag()/confusion_matrix.sum(1)).numpy(), 2)
                print('\n\ntrain Confusion Matrix: \n{}\n'.format(confusion_matrix_train.numpy().astype(np.int16)))
                print('\n\ntest Confusion Matrix: \n{}\n'.format(confusion_matrix.numpy().astype(np.int16)))


            plt.subplot(2,1,1)
            if epoch == 0 :
                plt.plot(epoch_list, val, 'red', label='loss')
            else :
                plt.plot(epoch_list, val, 'red')
      
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.title('loss graph')
            plt.legend(loc='upper right')

            plt.subplot(2,1,2)
            if epoch == 0:
                plt.plot(epoch_list, acc, 'blue', label='acc')
            else :
                plt.plot(epoch_list, acc, 'blue')

            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            plt.title('accuracy graph')
            plt.legend(loc='upper right')
            plt.savefig('{}/graph.png'.format(save_dir), dpi=300)

            print(classification_report(y_true, y_pred, target_names=target_name))
            #auroc = roc_auc_score(y_true, y_score, multi_class='ovr')
            f1_score = precision_recall_fscore_support(y_true, y_pred, average='macro')[2]
            f1_score = round(f1_score,5)
            save_acc = str(round(test_acc, 2)).split('.')
            save_acc = save_acc[0] + save_acc[1]

            save_name = "{}/".format(save_dir) + args.model_name+'_'+"f1_score"+ "_" + str(f1_score) + '_' + "acc"+ "_" + save_acc + '_' + "epoch" + str(epoch) + '.pt'
            torch.save(model.state_dict(), save_name)
           