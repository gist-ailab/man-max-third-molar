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
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epoch_num', type=int, default=1)
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
parser.add_argument('--weight', type=str, default='./weight/mandi_com/PNG+mask/checkpoint_model_ViT_large_r50_s32_384_mask_lr_9e-05_batchsize_8/ViT_large_r50_s32_384_mask_f1_score_0.73389_acc_7968_epoch45.pt') # 0.000009

#CBAM
parser.add_argument('--depth', type=int, default=50, choices=[18,34,50,101])
parser.add_argument('--att_type', type=str, default='CBAM', choices=['CBAM','BAM','basic'])
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
test_dataset = loader_down.DownDataLoader_com(test_dir, set="test", he=args.he_type, transform=transform2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle= False, num_workers=1)   

if args.model == 'Vision_Transformer':
    #model = timm.models.vision_transformer.vit_base_patch16_224_in21k(pretrained=True, num_classes=3)
    #model = timm.models.vision_transformer_hybrid.vit_base_r50_s16_224_in21k(pretrained=False, num_classes=3)
    #model = timm.models.vision_transformer_hybrid.vit_large_r50_s32_224_in21k(pretrained=False, num_classes=3)
    model= timm.models.vision_transformer_hybrid.vit_large_r50_s32_384(pretrained=False, num_classes=3)

elif args.model == 'resnet34':
    model = timm.models.resnet.resnet34(pretrained=False, num_classes=3)
elif args.model == 'resnet152d':
    model = timm.models.resnet.resnet152d(pretrained=False, num_classes=3)
elif args.model == 'seresnet101':
    model = timm.models.resnet.seresnet101(pretrained=False, num_classes=3)
elif args.model == 'efficientnet_b3':
    model = timm.models.efficientnet.efficientnet_b3(pretrained=False, num_classes=3)
elif args.model == 'coat':
    model = timm.models.coat.coat_lite_small(pretrained=False, num_classes=3)
model.load_state_dict(torch.load(args.weight))
model.cuda()

criterion = nn.CrossEntropyLoss()


model.eval()
val = []
acc = []

y_true = []
y_pred = []
y_score = []
target_name = ['very easy', 'easy', 'hard']
epoch = 0
with torch.no_grad():
    test_total = 0
    test_correct = 0
    test_bar = tqdm(test_loader)
    confusion_matrix_val = torch.zeros(3, 3)
    for test_iter, (images, label) in enumerate(test_bar):

        images = Variable(images).cuda()
        label = Variable(label).cuda()
        output, prob = model(images)

        _, predicted = torch.max(output.data, 1)
        ################## label edit #####################
        # print("label: {}".format(label))
        # print("softmax: {}".format(prob))
        # print("pred argmax : {}".format(torch.max(output.data, 1)[1]))
        # softmax = prob.cpu().numpy()

        # if label.item() != predicted.item():
        #     f = open("./mandibular_ian_label.txt", 'a')
        #     data = "img_name: {}, label: {}, pred: {}, softmax: {}\n".format(img_name[0], label.item(), predicted.item(), softmax[0])
        #     f.write(data)
        #     f.close
            
        ################ROC CURVE###################
        y_onehot = np.zeros((label.cpu().detach().numpy().shape[0], 3)).astype('uint8')
        y_onehot[np.arange(label.cpu().detach().numpy().shape[0]), label.cpu().detach().numpy()] = 1
    
        if test_iter <= 0:
            y_onehot_ = y_onehot[0]
            y_score_ =  prob[0].cpu().numpy()
        elif test_iter > 0:   
            y_onehot_ = np.vstack((y_onehot_, y_onehot[0]))
            y_score_ = np.vstack((y_score_, prob[0].cpu().numpy()))

        test_loss = criterion(output, label)
        test_total += label.size(0)
        test_correct += (predicted == label).sum().item()

        y_true.append(label.item())
        y_pred.append(predicted.item())

        y_score.append(prob[0].cpu().numpy())

        for t, p in zip(label.view(-1), predicted.view(-1)):
            confusion_matrix_val[t.long(), p.long()] += 1

        test_acc = 100 * test_correct / test_total

        test_bar.set_description(desc='[%d/%d]  Loss: %.4f Accuracy: %.2f' %
                                        (epoch, args.epoch_num - 1, test_loss.item(), test_acc))

    ################ROC CURVE###################
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_[:,i], y_score_[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_.ravel(), y_score_.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    fpr["macro"], tpr["macro"], _ = roc_curve(y_onehot_.ravel(), y_score_.ravel())
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(3):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= 3

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    plt.figure()

    plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)
    label_ = ['Low', 'Medium', 'High']
    for i in range(3):
        plt.plot(fpr[i], tpr[i], label='ROC curve of {0} (area = {1:0.2f})'
                                    ''.format(label_[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig("./auroc/down_com_roc.png", dpi=1000)
    
    per_class_acc = np.round((100 * confusion_matrix_val.diag()/confusion_matrix_val.sum(1)).numpy(), 2)
    print('\n\ntest Confusion Matrix: \n{}\n'.format(confusion_matrix_val.numpy().astype(np.int16)))
    print(classification_report(y_true, y_pred, target_names=target_name))
    ######################AUROC#############################
    print("auc roc : ", roc_auc_score(y_true, y_score,  average='macro', multi_class='ovr'))
    ################## sensitivity specificity ######################
    res = []
    sen = []
    spe = []
    for l in [0,1,2]:
        prec,recall,_,_ = precision_recall_fscore_support(np.array(y_true)==l, np.array(y_pred)==l, pos_label=True, average=None)
        res.append([l, recall[0], recall[1]])
        sen.append(recall[0])
        spe.append(recall[1])
    print("sensitivity average: {} | specificity: {}".format(np.mean(np.array(sen)), np.mean(np.array(spe))))
    pd.DataFrame(res, columns = ['class', 'sensitivity','specificity'])
    print(pd.DataFrame(res, columns = ['class', 'sensitivity','specificity']))
    ########################f1 score #########################
    f1_score = precision_recall_fscore_support(y_true, y_pred, average='macro')[2]
    f1_score = round(f1_score,4)
    print("f1 score : ", f1_score)

    #########confusion matrix############
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
    class_names = ['Low', 'Medium', 'High']
    cnf_matrix = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')

    plt.savefig("./auroc/CM_IAN.png", dpi=1000)

    # np_confusion_matrix = np.array(confusion_matrix)
    # con = np_confusion_matrix.astype(int)
    # fig ,ax = plot_confusion_matrix(con, colorbar=True, show_absolute=True, show_normed=True, figsize = (50.0,50.0))
    # plt.savefig("./auroc/CM_IAN.png", dpi=600)

    # from sklearn.metrics import confusion_matrix, plot_confusion_matrix
    # import matplotlib.pyplot as plt

    # label_ = ['N.1', 'N.2', 'N.3']
    # plot = plot_confusion_matrix(confusion_matrix, y_pred, y_true, display_labels=label_, cmap=None ,normalize=None)
    # plot.ax_.set_title('Confusion Matrix')
    # plt.savefig("./auroc/CM_IAN.png", dpi=900)
