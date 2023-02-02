# Automated Diagnosis for Extraction Difficulty of Maxillary and Mandibular Third Molars and Post-Extraction Complications Using Deep Learning (Submit: Journal of dentistry)

## overview

https://github.com/gist-ailab/man-max-third-molar/files/10543329/main_figure_V6.pdf

## Getting Started

### Environment Setup

RTX 2080ti with python 3.8, pytorch 1.9.0, torchvision 0.10.0 and CUDA 11.3


1. Set up a python environment
```
conda create -n third-molar python=3.8
conda activate third-molar
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -U openmim
mim install mmcv-full
pip install mmsegmentation

pip install tqdm
pip install albumentations
```

## Train & Evaluation

### Dataset Preparation
1. Download `third-molar dataset' from MAT.
```
segmentation location : ailab_mat/dataset/Third_molar/mmsegmentation/
classification location : ailab_mat/dataset/Third_molar/classification/
```

2. Organize the folders as follows
```
third_molar(segmentation)
├── leftImg8bit
       └── train
              └── 19760000.jpg.png
              ...
              └── 4182620000.jpg.png
       └── val
       └── test

├── gtFine
       └── train
              └── 19760000.jpg.png
              ...
              └── 4182620000.jpg.png
       └── val
       └── test

```
```
third_molar(classification)
├── crop_PNG_Images
       └── train
              └── 19760000.jpg.png
              ...
              └── 4182620000.jpg.png
       └── val
       └── test

├── crop_MASK_Images
       └── train
              └── 19760000.jpg.png
              ...
              └── 4182620000.jpg.png
       └── val
       └── test
├── Annotations
       └── 19760000.jpg.png.json
              └── "classTitle":"#48"
                  {"tags":[{"name":"P.III".......},
                            "name":"P.B".....}
                            "name":"N.3"....}
                            .
                            .
                            .
                            }
```
### Train on segmentation&classification dataset
```
CUDA_VISIBLE_DEVICES=0,1 sh mmsegmentation/tools/dist_train.sh ${CONFIG_FILE} 2
```
```
python classification/train_down_ext.py --gpu_id 0 --model Vision_Transformer --learning_rate 0.00001 --model_name ViT_large_r50_s32_384 --batch_size 8
```

### Evaluation on test dataset
```
python mmsegmentation/tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
```
```
python classification/test_down_ext.py --weight ${CHECKPOINT_FILE} 
```
### inference on test dataset
```
python demo/inference_demo.py
```

## License

The source code of this repository is released only for academic use. See the [license](./LICENSE.md) file for details.



## Authors
- **Junseok Lee, Jumi Park, Seong-yong Moon, Kyoobin Lee**

## License
Distributed under the MIT License.

## Acknowledgments
This work was supported by the ICT R&D program of MSIT/IITP[2020-0-00857, Development of Cloud Robot Intelligence Augmentation, Sharing and Framework Technology to Integrate and Enhance the Intelligence of Multiple Robots]
