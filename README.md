# ResNet model for image classification
## Description of the model
This repository contains the implementation of the **ResNet** model (460,842 parameters) for image classification (mnist dataset). The model is implemented using the Keras library.

Residual Neural Network (also known as Residual Network, ResNet ) is a deep learning model in which weight layers learn residual functions with reference to layer inputs. The architecture of such networks avoids the fading/exploding gradient, improves BackProp patency for large/huge convolutional models.

![ResBlock](https://github.com/Ultrageopro1966/ResNet/assets/120571667/61ab44e5-f592-476a-8794-a7c8c400f1ac)

> ResNet building block architecture

## Repository structure
- `Layers.py` - A file that defines the ResidualConvBlock and ResidualIdentitiBlock classes that are used in the ResNet model.
- `ResNet.py` - ResNet class definition file, which is a ResNet model for image classification.
- `DataPreparation.py` - A file with data preprocessing and preparation of datasets for training and testing the model.
- `Learing.py` - A file with training a ResNet model on pre-trained data and visualization of training results.
- `ResNet_Weights.h5` - The file where the trained weights of the model are saved.
- `configs.ini` - File containing model settings
## Usage
Install the required dependencies listed in `requirements.txt`

Run Learning.py to train the model on the pretrained data and save the trained weights to the ResNet_Weights.h5 file.
The model can be used to classify images by loading weights from ResNet_Weights.h5 and applying them to new data.

## Result of data preparation/training
![mnist](https://github.com/Ultrageopro1966/ResNet/assets/120571667/879c5d67-8258-44ba-8894-eed421454b53)

> Sample images of numbers

![AccuracyGraph](https://github.com/Ultrageopro1966/ResNet/assets/120571667/fc9862ea-51eb-4ff6-a2a5-321f92b19b89)

> Model training outcome (21st in the world according to [paperswithcode](https://paperswithcode.com/sota/image-classification-on-mnist?metric=Accuracy))

## Dependencies
- Tensorflow 2.x
- NumPy
- matplotlib

## Author
[UltraGeoPro](https://github.com/Ultrageopro1966)

## License
The ResNet model and code in this repository are available under the MIT license. See the LICENSE file for details.
