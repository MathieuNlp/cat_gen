# Cat Generator using DCGAN
The purpose of this project is generation cats images. To do so, I used a GAN model, specificaly a DCGAN model.
The DCGAN generator takes a vector z from the latent space and uses transpose convolution to map the features and output an image of size 3x64x64. The discriminator uses the same procedure but inverted.
![image](https://github.com/MathieuNlp/cat_gen/assets/78492189/432ddd64-f67f-426e-bee0-0184461d1866)

I used a public dataset available here: https://www.kaggle.com/datasets/crawford/cat-dataset

The model was trained on 225 epochs with a dataset of size around 9000 images. 

Modification were added in the training and dataset to avoid collapse mode:
- Preprocessing of the data: Center around the face of the cat and crop the dataset to 64x64 size (Highly helped the model to converge)
- Change of loss function from log(1-D(G(z))) to -log(D(G(z))) for the generator part (thanks to https://github.com/soumith/ganhacks)
## Setup
```sh
pip install -r requirements.txt
```
## Training
To run the training
```sh
python ./src/train_dcgan.py
```
## Generate samples from the generator
To generate 64 cats from dcgan

In config.yaml, choose a checkpoint for the generator (default is "./saved_model/generator_checkpoint_225.pt")
and run
```sh
python ./src/generate_fake.py
```
The generation plot should be saved in ./plots folder as sample_from_generator.png

## Source
Work greatly inspired by for training: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

For preprocessing: https://github.com/AlexiaJM/Deep-learning-with-cats

## Author
Mathieu Nalpon
