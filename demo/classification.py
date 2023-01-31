import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch
from PIL import Image
import cv2
import timm
from albumentations import CLAHE
import numpy as np
import albumentations

def loader(rgb, mask):
    clahe = CLAHE(clip_limit=(4.0,4.0), tile_grid_size=(8,8), p=1.0)
    data = {"image": rgb}
    he_rgb_img = clahe(**data)
    he_rgb_img = he_rgb_img["image"]
    
    transform = albumentations.Compose([
                albumentations.Resize(384,384),
                ])

    transform_rgb = transforms.Compose([
            #transforms.Resize((384,384)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ])
    transform_mask = transforms.Compose([
            #transforms.Resize((384,384)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485,], std = [0.229,])
            ])
    mask = mask.squeeze(2)
    augmented = transform(image = he_rgb_img, mask = mask)
    image = augmented["image"]
    mask = augmented["mask"]
    image = transform_rgb(Image.fromarray((image*255).astype(np.uint8)))
    mask = transform_mask(Image.fromarray((mask*255).astype(np.uint8)))
    image = torch.cat((image, mask),0)
    
    return image

# maxillary extraction difficulty prediction
def maxil_ext(rgb, mask):
    image = loader(rgb, mask)
    model = timm.models.vision_transformer_hybrid.vit_large_r50_s32_384(pretrained=False, num_classes=3)
    model.load_state_dict(torch.load("/data2/JSLEE/Third_molar/demo/checkpoint/maxil_ext/checkpoint_model_ViT_large_r50_s32_384_mask_2_lr_5e-05_batchsize_8/ViT_large_r50_s32_384_mask_2_f1_score_0.87749_acc_8797_epoch17.pt"))
    model.cuda()
    image = image.unsqueeze(0)
    image = image.cuda()
    output, prob = model(image)
    _, predicted = torch.max(output.data, 1)
    if predicted.item() == 0:
        ex_di = "vertical eruption"
    elif predicted.item() == 1:
        ex_di = "soft tissue impaction"
    elif predicted.item() == 2:
        ex_di = "complete bony impaction"
    else:
        pass

    return ex_di


def maxil_com(rgb, mask):
    image = loader(rgb, mask)
    model = timm.models.vision_transformer_hybrid.vit_large_r50_s32_384(pretrained=False, num_classes=3)
    model.load_state_dict(torch.load("/data2/JSLEE/Third_molar/demo/checkpoint/maxil_com/checkpoint_model_ViT_large_r50_s32_384_mask_lr_9e-05_batchsize_8_0/ViT_large_r50_s32_384_mask_f1_score_0.7716_acc_9014_epoch140.pt"))
    model.cuda()
    image = image.unsqueeze(0)
    image = image.cuda()
    output, prob = model(image)
    _, predicted = torch.max(output.data, 1)
    if predicted.item() == 0:
        si_per = "low"
    elif predicted.item() == 1:
        si_per = "medium"
    elif predicted.item() == 2:
        si_per = "high"
    else:
        pass

    return si_per


def mandi_ext(rgb, mask):
    image = loader(rgb, mask)
    model = timm.models.vision_transformer_hybrid.vit_large_r50_s32_384(pretrained=False, num_classes=4)
    model.load_state_dict(torch.load("/data2/JSLEE/Third_molar/demo/checkpoint/mandi_ext/checkpoint_model_ViT_large_r50_s32_384_mask_3_lr_8e-05_batchsize_8/ViT_large_r50_s32_384_mask_3_f1_score_0.79039_acc_8885_epoch89.pt"))
    model.cuda()
    image = image.unsqueeze(0)
    image = image.cuda()
    output, prob = model(image)
    _, predicted = torch.max(output.data, 1)
    
    if predicted.item() == 0:
        ex_di = "vertical eruption"
    elif predicted.item() == 1:
        ex_di = "soft tissue impaction"
    elif predicted.item() == 2:
        ex_di = "partial bony impaction"
    elif predicted.item() == 3:
        ex_di = "complete bony impaction"
    else:
        pass

    return ex_di

def mandi_com(rgb, mask):
    image = loader(rgb, mask)
    model = timm.models.vision_transformer_hybrid.vit_large_r50_s32_384(pretrained=False, num_classes=3)
    model.load_state_dict(torch.load("/data2/JSLEE/Third_molar/demo/checkpoint/mandi_com/checkpoint_model_ViT_large_r50_s32_384_mask_lr_6e-05_batchsize_8/ViT_large_r50_s32_384_mask_f1_score_0.83716_acc_8847_epoch126.pt"))
    model.cuda()
    image = image.unsqueeze(0)
    image = image.cuda()
    output, prob = model(image)
    _, predicted = torch.max(output.data, 1)
    if predicted.item() == 0:
        IAN_injury = "low"
    elif predicted.item() == 1:
        IAN_injury = "medium"
    elif predicted.item() == 2:
        IAN_injury = "high"
    else:
        pass

    return IAN_injury