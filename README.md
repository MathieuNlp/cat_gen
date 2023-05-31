# Cat Generator

## Folder layout

    ├── plots                         #Folder containing plots fake data vs real data and loss function
    ├── saved_model                   #Folder containing saved model where 5 means trained on 50 epochs
    ├── README.md
    ├── gan_model                    # classes of generator and discriminator
    ├── dataloader.py                # setup dataloader
    └── main.py                      #train and group all the pieces


## Dataset
Donwload dataset
```sh
https://www.kaggle.com/datasets/crawford/cat-dataset
```
## Training
To run the training
```sh
python ./main.py
```
## Generate fake samples
To generate 64 cats from dcgan
```sh
python ./generate_fake.py
```

## Source
Work greatly inspired by: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html


## Author
Mathieu Nalpon
