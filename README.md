# Temporal CycleGAN

The goal is to create a temporally consistent CylceGAN, based on [Youngwoon Lee's](https://github.com/youngwoon) implementation of a CycleGAN in Tensorflow.

## Description

```
![paper-figure](assets/paper-figure.png)
```

## Dependencies

- Python 3.6
- [Tensorflow 1.5.0](https://www.tensorflow.org/)
- [NumPy](https://pypi.python.org/pypi/numpy)
- [SciPy](https://pypi.python.org/pypi/scipy)
- [Pillow](https://pillow.readthedocs.io/en/4.0.x/)
- [tqdm](https://github.com/tqdm/tqdm)
- OpenCV

## Usage

- Execute the following command to train a model:

```
$ python train.py --task videos
```

- To reconstruct 256x256 images, set `--image_size` to 256; otherwise it will resize to and generate images in 128x128.
  Once training is ended, testing images will be converted to the target domain and the results will be saved to `./results/apple2orange_2017-07-07_07-07-07/`.
- Available datasets: apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos


- Check the training status on Tensorboard:

```
$ tensorboard --logdir=./logs
```



> **Carefully check Tensorboard for the first 1000 iterations. You need to run the experiment again if dark and bright regions are reversed like the exmaple below. This GAN implementation is sensitive to the initialization.**

![wrong-example](assets/wrong-initialization.png)

### Inference

Very much WIP, only single image A->B inference for now.

Example:

``
python test.py --input "test_image.jpeg" --output "test_output.jpg" --model_dir "logs/videos_2018-12-01_16-45-08"
``

## Results

TODO

## References

- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)
- The official implementation in Torch: https://github.com/junyanz/CycleGAN
  - The data downloading script is from the author's code.

