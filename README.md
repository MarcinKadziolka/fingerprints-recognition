## Installation
~~~
git clone git@github.com:MarcinKadziolka/done.git
cd done
conda env create -f environment.yml
conda activate fingerprints_recognition
~~~
## How to run
Download the dataset from [here](https://www.kaggle.com/datasets/ruizgara/socofing).


Create folder 'dataset' and place SOCOFing folder in it.

Folder structure should look like this:
~~~
database/
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
After training is finished you can find saved model at `wandb/latest_run/files/model.pth`
To evaluate:
~~~
python3 test.py pretrained/model.pth pretrained/config.yaml
~~~
