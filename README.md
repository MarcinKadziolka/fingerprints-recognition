# Fingerprints recognition using siamese network

This project focuses on recognizing fingerprints using a Siamese network. It leverages the Sokoto Coventry Fingerprint Dataset (SOCOFing) and applies various preprocessing and augmentation techniques to enhance model robustness. The model is trained to differentiate between genuine and altered fingerprints, achieving high accuracy through meticulous preprocessing and augmentation steps. The provided instructions cover environment setup, dataset preparation, training, evaluation, and visualization of results.

<img src="https://github.com/MarcinKadziolka/fingerprints_recognition/assets/30349386/13bfc0a2-5287-4b3f-8184-e4d448460346" width="500">
<img src="https://github.com/MarcinKadziolka/fingerprints_recognition/assets/30349386/b1c6c6a3-c19b-4225-9586-330603b4bd86" width="500">
<img src="https://github.com/MarcinKadziolka/fingerprints_recognition/assets/30349386/7d2cb37e-a8a6-464b-83c1-b5ac43d87339" width="500">
<img src="https://github.com/MarcinKadziolka/fingerprints_recognition/assets/30349386/b9c8f63b-fbbb-4c1d-934a-3f1a1694f972" width="500">


## Installation
~~~
git clone git@github.com:MarcinKadziolka/fingerprints_recognition.git
cd fingerprints_recognition
conda env create -f environment.yml
conda activate fingerprints_recognition
~~~
## How to run
Download the dataset from [here](https://www.kaggle.com/datasets/ruizgara/socofing).


Create folder `dataset`:
~~~
mkdir dataset
~~~

Place `SOCOFing` folder in it. Folder structure should look like this:
~~~
dataset/
├── SOCOFing/                                                                                  
│   ├── Altered/
│   │   └── Altered-Easy
│   │   └── Altered-Medium
│   │   └── Altered-Hard                                                                                                                        
│   ├── Real/                                                                            
~~~

Create csv files:
~~~
python3 create_csv_from_images.py
~~~
Preprocess:
~~~
python3 preprocess.py
~~~
To train new model:
~~~
python3 train.py config.yaml
~~~
After training is finished you can find saved model at `wandb/latest_run/files/model.pth`.

To evaluate:
~~~
python3 test.py pretrained/model.pth pretrained/config.yaml
~~~

To visualize predictions:
~~~
python3 visualize_preds.py pretrained/model.pth pretrained/config.yaml
~~~

## Dataset
The Sokoto Coventry Fingerprint Dataset (SOCOFing) is a biometric database of fingerprints designed for academic research purposes. SOCOFing consists of 6000 fingerprint images from 600 individuals and includes unique attributes such as gender labels, hand and finger names, as well as synthetically altered versions of fingerprints with three different levels of modification: obliteration, central rotation, and z-cut.


<img src="https://github.com/MarcinKadziolka/fingerprints_recognition/assets/30349386/20e7a7ec-f266-4d2d-8bc5-f562e7781d43" width="600">

### Preparing dataset for siamese network
To create an appropriate dataset for training the Siamese network model, a process of generating fingerprint pairs was conducted. This process involved pairing each genuine fingerprint image with its three modifications (obliteration, z-cut, central rotation). Additionally, negative pairs were created by pairing any fingerprints while maintaining hand and finger compatibility.

## Preprocessing
- Trimming Black Pixels: All images had non-fingerprint pixels at the edges, especially visible after rotation augmentations introducing unnecessary noise. All images were thus trimmed to remove these pixels.
- Conversion to Grayscale: All images were converted to grayscale, reducing the number of channels from three to one, simplifying calculations.

<img src="https://github.com/MarcinKadziolka/fingerprints_recognition/assets/30349386/5a76aa61-0af6-4659-858f-5eb5285e7934" width="600">

## Augmentations
To enhance the model's robustness and improve its generalization capabilities, various data augmentation techniques were applied:

- Rotation: Involves rotating fingerprint images at various angles, enabling the model to recognize fingerprints from different angles.
- Gaussian Blur: Applying Gaussian Blur introduces a level of blurriness, helping the model deal with minor fluctuations and irregularities.
- Shear: Distorts images by shifting them along the horizontal or vertical axis, teaching the model to recognize fingerprints at various angles and bends.
- Random Resizing: Involves scaling images at different ratios, allowing the model to recognize fingerprints of varying sizes, useful for different scanner types.
- Random Perspective: Randomly modifies the image perspective, simulating different viewing angles, useful for fingerprints registered at various angles.

These augmentation techniques expose the model to a greater variety of training data, leading to better performance under diverse conditions and unconventional cases, thereby enhancing its overall resilience and fingerprint recognition capabilities in various scenarios.

## Results
The model was trained for 20 epochs using Binary Cross-Entropy (BCE) as the loss function and the Adam optimizer with a learning rate of 0.001. As the model outputs values between zero and one (similarity of fingerprints), a threshold needs to be chosen for categorizing predictions. During training and validation, the threshold was set to 0.9. 

<img src="https://github.com/MarcinKadziolka/fingerprints_recognition/assets/30349386/1eebbe5b-8ec9-40e3-9839-8f469e268dad" width="900">
<img src="https://github.com/MarcinKadziolka/fingerprints_recognition/assets/30349386/ecf354de-04fa-4de3-91c3-d98a63a8c3aa" width="900">

During testing with threshold of 0.5, model achieved accuracy above 90%.

<img src="https://github.com/MarcinKadziolka/fingerprints_recognition/assets/30349386/03b7f12e-9a1c-4e6b-b02e-4980d1b01e75" width="500">

