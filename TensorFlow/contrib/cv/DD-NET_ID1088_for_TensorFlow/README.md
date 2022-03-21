# DD-Net 
(A Double-feature Double-motion Network)
[![](mics/paper.png)](https://arxiv.org/pdf/1907.09658.pdf)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/make-skeleton-based-action-recognition-model-1/skeleton-based-action-recognition-on-jhmdb-2d)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-jhmdb-2d?p=make-skeleton-based-action-recognition-model-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/make-skeleton-based-action-recognition-model-1/skeleton-based-action-recognition-on-shrec)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-shrec?p=make-skeleton-based-action-recognition-model-1)


# About this code
This repo contains code that can run on Ascend 910, which is convered from the original repo: https://github.com/fandulu/DD-Net. For datasets, only JHMDB is supported currently.

# How to run this code
Data preparation: 

You can get the JHMDB dataset from the original repo on GitHub, and put the relevant files in data/JHMDB.

Running:
1. Create a notebook environment with Ascend 910 on ModelArts.
2. Upload this folder to the work dir.
3. Copy the commands in run.txt, and run in jupyter notebook.

# Performance
Pricision:
|Dataset | On GPU | On Ascend 910 |
| :----: | :----: | :----: |
| JHMDB | 83.52 | 84.09  |

Speed:
| Platform | On GPU | On Ascend 910 |
| :----: | :----: | :----: |
| Avarage latancy per sample | about 200 us | 285 us |


