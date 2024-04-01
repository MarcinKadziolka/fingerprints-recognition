# Fingerprints recognition using siamase network

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
## Dataset
The Sokoto Coventry Fingerprint Dataset (SOCOFing) is a biometric database of fingerprints designed for academic research purposes. SOCOFing consists of 6000 fingerprint images from 600 individuals and includes unique attributes such as gender labels, hand and finger names, as well as synthetically altered versions of fingerprints with three different levels of modification: obliteration, central rotation, and z-cut.


<img src="https://github.com/MarcinKadziolka/fingerprints_recognition/assets/30349386/20e7a7ec-f266-4d2d-8bc5-f562e7781d43" width="600">

