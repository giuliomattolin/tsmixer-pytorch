# TSMixer
PyTorch Implementation of ["TSMixer: An All-MLP Architecture for Time Series Forecasting"](https://arxiv.org/abs/2303.06053)

## Installation
Install the dependencies:
```
pip install -r requirements.txt
```

## Data Preparation
We use pre-processed datasets provided in [Autoformer](https://github.com/thuml/Autoformer).
```
mkdir dataset
cd dataset
# Download zip file from [Google Drive](https://drive.google.com/corp/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) and put it under dataset/
unzip all_six_datasets.zip
mv all_six_datasets/*/*.csv ./
```

## Training Example
To reproduce results of 96 prediction length run the following bash scripts:

**ETTm2**
```
python main.py --data <path_to_csv_file> --seq_len 512 --pred_len 96 --learning_rate 0.001 --n_block 2 --dropout 0.9 --ff_dim 64
```
**Weather**
```
python main.py --data <path_to_csv_file> --seq_len 512 --pred_len 96 --learning_rate 0.0001 --n_block 4 --dropout 0.3 --ff_dim 32
```
**Electricity**
```
python main.py --data <path_to_csv_file> --seq_len 512 --pred_len 96 --learning_rate 0.0001 --n_block 4 --dropout 0.7 --ff_dim 64
```
**Traffic**
```
python main.py --data <path_to_csv_file> --seq_len 512 --pred_len 96 --learning_rate 0.0001 --n_block 8 --dropout 0.7 --ff_dim 64
```
