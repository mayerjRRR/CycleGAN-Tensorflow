# Temporally Coherent CycleGAN

The goal is to create a temporally coherent CylceGAN. 

<p align = 'center'>
<img src = 'results/horse.gif' width = '49%'>
<img src = 'results/zebra_temp.gif' width = '49%'>
</p>

The implementation is based on [Youngwoon Lee's implementation](https://github.com/gitlimlab/CycleGAN-Tensorflow) of a CycleGAN in Tensorflow.

## Installation 

### Prerequisites:

- a working Python 3.6.x installation
- a working CUDA installation for tensorflow 1.8.0 
- optionally a virtual environment

### Installation:

- clone the repository 
- ``pip install -r requirements.txt``

## Usage

### Training

- Execute the following command to train a model:

```
$ python train.py --task <taskname>
```

The corresponding `./dataset/<taskname>` directory should contain `trainA` and `trainB` directories containing training
data for domain A and domain B respectively. 

The training data can be either images or videos for each domain. Handling of different data types is done automatically.

Command line options:

- `--task` : Name of the task, specifies the training data directory
- `--image_size`: Resolution of the training data, default is 256
- `--load_model`: Optional log/checkpoint directory of an existing model, use to continue training

More command line options can be found with `--help`.

Once training, check the status on Tensorboard:

```
$ tensorboard --logdir=./logs
```


### Inference

Applies a domain transfer using a trained model to an input. Works for images and videos in both A->A an B->A directions. 

Example:

``
python test.py --input "test_image.jpeg" --output "test_output.jpeg"
``

Command line options:

- `--input` : Name of the input file, defaults to test_image.jpeg`.
- `--output`: Name of the output file, media type should match input type. Note that sound is lost for video transfer. 
Defaults to `test_output.jpeg`
- `--model_dir`: The pretrained model to use (name of the model directory). If none is given it will automatically use 
the latest trained model.
- `--backwards`: Optional flag to perform B->A inference instead of the default A->B.

More command line options can be found with `--help`.

## Results

Example on the domains "Horse" and "Zebra" trained without (middle) and with temporal discriminator (right).

<p align = 'center'>
<img src = 'results/horse.gif' width = '99%'>
<img src = 'results/zebra_non_temp.gif' width = '99%'>
<img src = 'results/zebra_temp.gif' width = '99%'>
</p>

Low-Quality to High-Quality Render of Smoke Simulations, trained with images only (top) and temporal discriminator, recurrent generator and pingpong loss (bottom).

<p align = 'center'>
<img src = 'results/plume_images.gif' width = '99%'>
<img src = 'results/plume_recurrent.gif' width = '99%'>
</p>

