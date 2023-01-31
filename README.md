# Automated Diagnosis for Extraction Difficulty of Maxillary and Mandibular Third Molars and Post-Extraction Complications Using Deep Learning

## overview

https://github.com/gist-ailab/man-max-third-molar/files/10543329/main_figure_V6.pdf

## Getting Started

### Environment Setup

Tested on RTX 2080ti with python 3.8, pytorch 1.9.0, torchvision 0.10.0 and CUDA 11.3


1. Set up a python environment
```
conda create -n third-molar python=3.8
conda activate third-molar
conda install pytorch torchvision -c pytorch
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
segmentation location : ailab_mat/dataset/Third_molar/mmsegmentation/dataset
classification location : ailab_mat/dataset/Third_molar/classification/dataset
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
```
### Train on sample dataset
```
python main.py --gpu_num 0
```

### Evaluation on test dataset
```
python inference.py --gpu_num 0 --th 0.4
```

### Evaluation on test dataset
```
python visualization.py --gpu_num 0 --th 0.4
```

## License

The source code of this repository is released only for academic use. See the [license](./LICENSE.md) file for details.



## Authors
- **Jongwon Kim** [jongwonkim](https://github.com/jwk92)

## License
Distributed under the MIT License.

## Acknowledgments
This work was supported by the ICT R&D program of MSIT/IITP[2020-0-00857, Development of Cloud Robot Intelligence Augmentation, Sharing and Framework Technology to Integrate and Enhance the Intelligence of Multiple Robots]
