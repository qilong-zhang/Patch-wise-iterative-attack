# Patch-wise Iterative  Attack (accpeted by ECCV2020)
This is the **Pytorch** code for our paper [Patch-wise Attack for Fooling Deep Neural Network](http://arxiv.org/abs/2007.06765).  

:boom:Although we did not add experiments of targeted version in the paper, our patch-wise approach does improve the success rate by a large margin. We will add this part of the experiment later, and we hope that you could consider our algorithm when implementing targeted attacks:smile:.

In our paper, we propose a novel  **patch-wise iterative  Attack** by using the amplification factor and guiding gradient to its feasible direction. Comparing with state-of-the-art attacks, we further improve the success rate by 3.7\% for normally trained models and 9.1\% for defense models on average. We hope that the proposed methods will serve as a benchmark for evaluating the robustness of various deep models and defense methods.

## Implementation
- Pytorch 1.5.1, Python3.7 

- Dataset could be found in tensorflow version (../dataset/images) which is already downloaded.

- Run the code

  ```python
  python project_iter_attack.py --input_csv your_patch --input_dir your_path
  ```


## Results

![](https://github.com/qilong-zhang/patch-wise-iterative-attack/blob/master/readme_img/illustration.png)

![result](https://github.com/qilong-zhang/patch-wise-iterative-attack/blob/master/readme_img/result.png)



## Citing this work

If you find this work is useful in your research, please consider citing:

```
@inproceedings{Zhang2020PatchWise,
    title={Patch-wise Attack for Fooling Deep Neural Network},
    author={Gao, Lianli and Zhang, Qilong and Song, jingkuan and Liu, Xianglong and Shen, Hengtao},
    Booktitle = {European Conference on Computer Vision},
    year={2020}
}
```

