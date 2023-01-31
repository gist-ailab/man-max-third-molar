from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
from torch.autograd import Variable
import os
import numpy as np
import loader_down
import glob
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tqdm import tqdm
import time
import albumentations
import albumentations
import random
import timm
from imbalanced_dataset_sampler.torchsampler.imbalanced import ImbalancedDatasetSampler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from scipy import interp
import pandas as pd

matplotlib.use('Agg')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
print(time.strftime('%Y-%m-%d', time.localtime(time.time())))
# Learning
parser.add_argument("--learning_rate", default=3e-2, type=float,
                    help="The initial learning rate for SGD.")
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epoch_num', type=int, default=200)
parser.add_argument('--gpu_id', type=str ,default='4')
parser.add_argument('--model_name', type=str, default='resnet')
parser.add_argument("--model", choices=["Vision_Transformer", "resnet34", "resnet152d","seresnet", "efficientnet_b3", "R50-ViT-B_16", "coat"],
                    default="Vision_Transformer",help="Which model to use.")
parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16","ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                    default="R50-ViT-B_16",help="Which variant to use.")
parser.add_argument("--img_size", default=384, type=int, help="Resolution size")
parser.add_argument("--pretrained_dir", type=str, default="Vit_pretrain/imagenet21k/imagenet21k_ViT-B_16.npz", 
                    help="Where to search for pretrained ViT models.")
parser.add_argument("--weight_decay", default=0, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")
parser.add_argument('--he_type', type=str, default='CLAHE', choices=['CLAHE', 'HE'])
parser.add_argument('--weight', type=str, default='./weight/Paper_/down_ext/checkpoint_model_Resnet34_224_lr_5e-05_batchsize_16/Resnet34_224_f1_score_0.60741_acc_8435_epoch49.pt')
#parser.add_argument('--weight', type=str, default='./weight/weight3/Paper/down_ext_new/checkpoint_model_R50+ViT-B_16_384_ce_lr_1e-05_batchsize_8/R50+ViT-B_16_384_ce_f1_score_0.72587_acc_856_epoch97.pt') #lr 0.000005

args = parser.parse_args()


# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# set seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

transform2 = albumentations.Compose([
                albumentations.Resize(args.img_size,args.img_size),
                #albumentations.CLAHE(clip_limit=(3.0,4.0), tile_grid_size=(8,8), p=1.0)
                ])


test_dir = []
test_imgs = glob.glob("./dataset/crop_PNG_Images/test_man/*.png")
test_dir = test_imgs
test_dataset = loader_down.DownDataLoader_ext(test_dir, set="test", he=args.he_type, transform=transform2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle= False, num_workers=1)        

num_classes=4

#Vision Transformer 
if args.model == 'Vision_Transformer':
    #model = timm.models.vision_transformer.vit_base_patch16_224_in21k(pretrained=False, num_classes=4)
    #model = timm.models.vision_transformer_hybrid.vit_base_r50_s16_224_in21k(pretrained=False, num_classes=4)
    #model = timm.models.vision_transformer_hybrid.vit_large_r50_s32_224_in21k(pretrained=False, num_classes=4)
    model= timm.models.vision_transformer_hybrid.vit_large_r50_s32_384(pretrained=True, num_classes=4)

elif args.model == 'resnet34':
    model = timm.models.resnet.resnet34(pretrained=True, num_classes=4)
elif args.model == 'resnet152d':
    model = timm.models.resnet.resnet152d(pretrained=True, num_classes=4)
elif args.model == 'seresnet101':
    model = timm.models.resnet.seresnet101(pretrained=True, num_classes=4)
elif args.model == 'efficientnet_b3':
    model = timm.models.efficientnet.efficientnet_b3(pretrained=True, num_classes=4)
elif args.model == 'coat':
    model = timm.models.coat.coat_lite_small(pretrained=False, num_classes=4)
model.load_state_dict(torch.load(args.weight))
model.cuda()

criterion = nn.CrossEntropyLoss()



model.eval()
val = []
acc = []

y_true = []
y_pred = []
y_score = []
target_name = ['1', '2', '3', '4']
epoch = 0
with torch.no_grad():
    test_total = 0
    test_correct = 0
    test_bar = tqdm(test_loader)
    confusion_matrix = torch.zeros(4, 4)
    count = 366
    for test_iter, (images, label) in enumerate(test_bar):
     
        images = Variable(images).cuda()
        label = Variable(label).cuda()
        output, prob = model(images)

        _, predicted = torch.max(output.data, 1)
        ################## label edit #####################
        # print("label: {}".format(label_))
        # print("softmax: {}".format(prob))
        # print("pred argmax : {}".format(torch.max(output.data, 1)[1]))
        # softmax = prob.cpu().numpy()
        

        # if label.item() != predicted.item():
        #     f = open("./mandibular_label.txt", 'a')
        #     data = "img_name: {}, label:{} {}, pred: {}, softmax: {}\n".format(img_name[0], label.item(), label_, predicted.item(), softmax[0])
        #     f.write(data)
        #     f.close
        ################ROC CURVE###################
        y_onehot = np.zeros((label.cpu().detach().numpy().shape[0], 4)).astype('uint8')
        y_onehot[np.arange(label.cpu().detach().numpy().shape[0]), label.cpu().detach().numpy()] = 1
    
        if test_iter <= 0:
            y_onehot_ = y_onehot[0]
            y_score_ =  prob[0].cpu().numpy()
        elif test_iter > 0:   
            y_onehot_ = np.vstack((y_onehot_, y_onehot[0]))
            y_score_ = np.vstack((y_score_, prob[0].cpu().numpy()))
 
        ################ROC CURVE###################
        ###########################
        # if label.item() == 1:
        #     if predicted.item() == 1:
        #         f = open("/data_4/JSLEE/Wisdom_classification_v5/weight3/down_extraction_gt.txt",'a')
        #         data1 = "file_name:{}, pred:{}\n".format(str(img_data[0]) + ".png", predicted.item())
        #         f.write(data1)
        ###########################
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
    

    ################ROC CURVE###################
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_[:,i], y_score_[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_.ravel(), y_score_.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    fpr["macro"], tpr["macro"], _ = roc_curve(y_onehot_.ravel(), y_score_.ravel())
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(4)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(4):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= 4
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plt.figure()

    # plt.plot(fpr["micro"], tpr["micro"],
    #      label='micro-average ROC curve (area = {0:0.2f})'
    #            ''.format(roc_auc["micro"]),
    #      color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)
    label_ = ['VE', 'STI', 'PBI', 'CBI']
    for i in range(4):
        plt.plot(fpr[i], tpr[i], label='ROC curve of {0} (area = {1:0.2f})'
                                    ''.format(label_[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig("./auroc/down_ext_roc.png", dpi=1000)
    

    per_class_acc = np.round((100 * confusion_matrix.diag()/confusion_matrix.sum(1)).numpy(), 2)
    print('\n\ntest Confusion Matrix: \n{}\n'.format(confusion_matrix.numpy().astype(np.int16)))
    print(classification_report(y_true, y_pred, target_names=target_name))
    ################## sensitivity specificity ######################
    res = []
    sen = []
    spe = []
    for l in [0,1,2,3]: #####edit
    #for l in [0,1,2]:
        prec,recall,_,_ = precision_recall_fscore_support(np.array(y_true)==l, np.array(y_pred)==l, pos_label=True, average=None)
        res.append([l, recall[0], recall[1]])
        sen.append(recall[0])
        spe.append(recall[1])
    print("sensitivity average: {} | specificity: {}".format(np.mean(np.array(sen)), np.mean(np.array(spe))))
    pd.DataFrame(res, columns = ['class', 'sensitivity','specificity'])
    print(pd.DataFrame(res, columns = ['class', 'sensitivity','specificity']))
    #########################AUROC###############################
    print("auc roc : ", roc_auc_score(y_true, y_score,  average='macro', multi_class='ovr'))
    ##########################F1score#############################
    f1_score = precision_recall_fscore_support(y_true, y_pred, average='macro')[2]
    f1_score = round(f1_score,4)
    print("f1 score : ", f1_score)
        

    #########confusion matrix############
    # from sklearn.metrics import confusion_matrix, plot_confusion_matrix
    # import matplotlib.pyplot as plt

    # label_ ['VE','STI', 'PBI', 'CBI']
    # plot = plot_confusion_matrix(model, y_pred, y_true, display_labels=label_, cmap=None, normalize=None)
    # plot.ax_.set_title('Confusion Matrix')
    # plt.savefig("./auroc/CM_EXT.png", dpi=900)

    from sklearn.metrics import confusion_matrix
    import itertools

    def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix', cmap=plt.cm.Blues):

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    plt.figure()
    class_names = ['VE','STI', 'PBI', 'CBI']
    cnf_matrix = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')

    plt.savefig("./auroc/CM_EXT.png", dpi=1000)