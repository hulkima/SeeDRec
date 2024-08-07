# SeeDRec
The source code is for the paper: SeeDRec: Sememe-based Diffusion for Sequential Recommendation accepted in IJCAI 2024 by Haokai Ma, Ruobing Xie, Lei Meng, Yimeng Yang, Xingwu Sun and Zhanhui Kang.

## Overview
Inspired by the power of Diffusion Models (DM) verified in various fields, some pioneering works have started to explore DM in recommendation. However, these prevailing endeavors commonly implement diffusion on item indices, leading to the increasing time complexity, the lack of transferability, and the inability to fully harness item semantic information. To tackle these challenges, we propose SeeDRec, a sememe-based diffusion framework for sequential recommendation (SR). Specifically, inspired by the notion of sememe in NLP, SeeDRec first defines a similar concept of recommendation sememe to represent the minimal interest unit and upgrades the specific diffusion objective from the item level to the sememe level. With the Sememe-to-Interest Diffusion Model (S2IDM), SeeDRec can accurately capture the user’s diffused interest distribution learned from both local interest evolution and global interest generalization while maintaining low computational costs. Subsequently, an Interest-aware Prompt-enhanced (IPE) strategy is proposed to better guide each user’s sequential behavior modeling via the learned user interest distribution. Extensive experiments on nine SR datasets and four cross-domain SR datasets verify its effectiveness and universality.![_](./overall_structure.png)

## Dependencies
- Python 3.8.10
- PyTorch 1.12.0+cu102
- pytorch-lightning==1.6.5
- Torchvision==0.8.2
- Pandas==1.3.5
- Scipy==1.7.

## Dependencies
- Python 3.8.10
- PyTorch 1.12.0+cu102
- Torchvision==0.8.2
- Pandas==1.3.5
- Scipy==1.7.3

### SeeDRec (SASRec) on Home:
```
CUDA_VISIBLE_DEVICES=3 python SeeDRec.py --cross_dataset=Home --lr 0.005 --temperature 10 --index 001
```
### SeeDRec (SASRec) on Electronic:
```
CUDA_VISIBLE_DEVICES=3 python SeeDRec.py --cross_dataset=Electronic --lr 0.001 --temperature 10 --index 001
```

## BibTeX
If you find this work useful for your research, please kindly cite SeeDRec by:
```
@inproceedings{SeeDRec,
  title={SeeDRec: Sememe-based Diffusion for Sequential Recommendation},
  author={Ma, Haokai and Xie, Ruobing and Meng, Lei and Yang, Yimeng and Sun, Xingwu and Kang, Zhanhui},
  booktitle={Proceedings of the International Joint Conference on Artificial  Intelligence},
  year={2024}
}
```

## Acknowledgement
The structure of this code is largely based on [DiffRec](https://github.com/YiyanXu/DiffRec), [SASRec](https://github.com/pmixer/SASRec.pytorch) and [PDRec](https://github.com/hulkima/PDRec) and the dataset is collected by [Amazon](https://nijianmo.github.io/amazon/index.html). Thanks for these works.



