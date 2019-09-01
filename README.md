# TricycleGAN: A Temporally Consistent CycleGAN for Unpaired Video-to-Video Translation

<p align = 'center'>
<img src = 'results/smoke_512_tricycle.gif' width = '100%'>
</p>

This code in this repository was created as part of my (Jonas Mayer's) master's thesis at TU Munich. The actual thesis and supplementary material can be found down below.
Please steal responsibly. 

The implementation is based on [Youngwoon Lee's implementation](https://github.com/gitlimlab/CycleGAN-Tensorflow) of a CycleGAN in Tensorflow.

## Installation 

### Prerequisites:

- a working Python 3.6 installation
- a working CUDA installation for TensorFlow
- optionally a virtual environment

### Installation:

- clone the repository 
- ``pip install -r requirements.txt``

## Usage

### Training

- Execute the following command to train a model:

```
$ python train.py -n <run name> --task <task name>
```

Command line options:

- `--n`: name of the training run, useful for finding your training results later on. 
- `--task` : Name of the task, specifies the training data directory

More command line options can be found with `--help`.

The corresponding `./dataset/<taskname>` directory should contain `trainA` and `trainB` directories, each containing training
data for domain A and domain B respectively. 
The training data can be either images (*.jpg xor *.png) or videos (*.mp4). 
Depending on which type of data exists, we either train a cycleGAN or a tricycleGAN.
However, if video data is available as images, make sure to place it in ``trainA/frames/`` and ``trainA/frames/`` and use `--force_video True` to force video training.
For training a cycleGAN with video data, use ``--force_images True``.

Once training, check TensorBoard for live losses, metrics, output images and training parameters:

```
$ tensorboard --logdir=./logs
```

### Inference

Applies a domain transfer using a trained model to an input. Works for single images, image directories and videos in both A->A an B->A directions. 

Example:

``
python test.py --input "beautiful_horsies.mp4" --output "beautiful_zebras.mp4"
``

Command line options:

- `--input` : Name of the input file or directory, defaults to `smoke_input.jpg`.
- `--output`: Name of the output file. Note that sound is lost for video transfer. 
Defaults to `smoke_output.jpeg`
- `--model_dir`: The pretrained model to use (name of the model directory). If none is given it will automatically use 
the latest trained model.
- `--no_temp`: Use for inference with a cycleGAN model.

More command line options can be found with `--help`.

## Results

Check out the supplementary video on YouTube:

[![Click for youtube video](https://img.youtube.com/vi/L86PNqh_zLI/0.jpg)](https://www.youtube.com/watch?v=L86PNqh_zLI)

## Thesis, Presentation and Supplementary Material

https://drive.google.com/open?id=1fv_LD4Sva8Kcm9xwZABKzH-I2z0Ya2xO