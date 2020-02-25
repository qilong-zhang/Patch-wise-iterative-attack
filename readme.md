# Projection iterative  Attacks

## Implementation
- Tensorflow 1.14, Python3.7

- Download the model to attack

  - [Normlly trained models](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained)(DenseNet can be found in [here](https://github.com/flyyufelix/DenseNet-Keras))
  - [Ensemble  adversarial trained models](https://github.com/tensorflow/models/tree/master/research/adv_imagenet_models?spm=5176.12282029.0.0.3a9e79b7cynrQf)
  - [Feature  Denoise](https://github.com/facebookresearch/ImageNet-Adversarial-Training)

- Then put these models into ".models/"

- Run the code

  ```python
  python project_iter_attack.py
  ```

- The output images are in "output/"



## Results

![TIM截图20200225160731](TIM截图20200225160731.png)



### Citation

