# Instance-aware-RES

**Instance-aware Referring Expression Segmentation**

<div align="center">
  <img src="/framework.pdf"/>
</div><br/>

## Setup

We recommended the following dependencies.

* Python 3.6
* Numpy

This code is derived from [RRN](https://github.com/liruiyu/referseg_rrn) \[2\]. Please refer to it for more details of setup.

## Data Preparation
* Dataset Preprocessing

We conduct experiments on 4 datasets of referring image segmentation, including `UNC`, `UNC+`, `Gref` and `ReferIt`. After downloading these datasets, you can run the following commands for data preparation:
```
python build_batches.py -d Gref -t train
python build_batches.py -d Gref -t val
python build_batches.py -d unc -t train
python build_batches.py -d unc -t val
python build_batches.py -d unc -t testA
python build_batches.py -d unc -t testB
python build_batches.py -d unc+ -t train
python build_batches.py -d unc+ -t val
python build_batches.py -d unc+ -t testA
python build_batches.py -d unc+ -t testB
python build_batches.py -d referit -t trainval
python build_batches.py -d referit -t test
```

* Glove Embedding

Download `Gref_emb.npy` and `referit_emb.npy` and put them in `data/`. We provide download link for Glove Embedding here:
[Baidu Drive](https://pan.baidu.com/s/19f8CxT3lc_UyjCIIE_74FA), password: 2m28.


## Training
Train on UNC training set with:
```
python train.py --config-file configs/my_Model_soft2-unc-3x.yaml --num-gpus 4
```

## Testing
Test on UNC validation set with:
```
python train.py --config-file configs/my_Model_soft2-unc-3x.yaml --num-gpus 4 --eval-only MODEL.WEIGHTS ./output/my_Model_soft2_ab2/unc/1x/model_final.pth DATASETS.TEST \(\"unc_val\",\)
```
