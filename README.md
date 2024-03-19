
# Better “CMOS” Produces Clearer Images: Learning Space-Variant Blur Estimation for Blind Image Super-Resolution (CMOS, CVPR 2023)

This repository is the official PyTorch implementation of Better “CMOS” Produces Clearer Images: Learning Space-Variant Blur Estimation for Blind Image Super-Resolution
([arxiv](https://arxiv.org/abs/2304.03542)). 
Please feel free to contact us if you have any questions.


## Requirements
- Prepare experimental environments
  ```
  pip install -r requirements.txt
  ```

Note: this repository is based on [BasicSR](https://github.com/xinntao/BasicSR#memo-codebase-designs-and-conventions) and [MANet](https://github.com/JingyunLiang/MANet). Please refer to their repository for a better understanding of the code framework.


## Data Preparation
To prepare data, download datasets [NYUv2-BSR](https://drive.google.com/file/d/1W9zy45nvje8zQ7QaU0DK1c_9hPeGqB8R/view?usp=drive_link) and [Cityscapes-BSR](https://drive.google.com/file/d/1RLcLZbdq7qhqDgl4elywKSZX3Sq7Nwd3/view?usp=drive_link) from Google Drive and place them in the './datasets' folder. The creation of these two datasets is based on [this repository](https://github.com/codeslake/SYNDOF). Please refer to our paper for more detais.


## Training
Step1: to train CMOS with NYUv2-BSR, run this command:

```
python train.py --opt options/train/NYUv2_BSR/train_stage1.yml
```

Step2: to train non-blind RRDB-SFT with NYUv2-BSR, run this command:

```
python train.py --opt options/train/NYUv2_BSR/train_stage2.yml
```

Step3: to fine-tune RRDB-SFT with CMOS, run this command:

```bash
python train.py --opt options/train/NYUv2_BSR/train_stage3.yml
```

All trained models are provided in `./codes/pretrained_models`.


## Testing

To test CMOS (stage1, blur and semantic estimation) with NYUv2-BSR, run this command:

```
python test.py --opt options/test/NYUv2_BSR/test_stage1.yml
```
To test RRDB-SFT (stage2, non-blind SR with ground-truth blur and semantic maps) with NYUv2-BSR, run this command:

```
python test.py --opt options/test/NYUv2_BSR/test_stage2.yml
```
To test CMOS+RRDB-SFT (stage3, blind SR), run this command:

```
python test.py --opt options/test/NYUv2_BSR/test_stage3.yml
```

## Citation
    @inproceedings{chen2023better,
        title={Better" CMOS" Produces Clearer Images: Learning Space-Variant Blur Estimation for Blind Image Super-Resolution},
        author={Chen, Xuhai and Zhang, Jiangning and Xu, Chao and Wang, Yabiao and Wang, Chengjie and Liu, Yong},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        pages={1651--1661},
        year={2023}
    }

## Acknowledgement

The codes are based on [BasicSR](https://github.com/xinntao/BasicSR), [MANet](https://github.com/JingyunLiang/MANet) and [MTI-Net](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch). Thanks for their great works.



