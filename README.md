# Instance-aware-RES

**Instance-aware Referring Expression Segmentation**

Shijia Huang, Tiancheng Shen, Yanwei Li, Shu Liu, Jiaya Jia, Liwei Wang

[[`arXiv`](https://arxiv.org/pdf/xxxx.xxxx.pdf)] [[`BibTeX`](#CitingInstNet)]

<div align="center">
  <img src="/framework.png"/>
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

## Pre-trained Models and Logs

We provide the [pre-trained models](https://www.abc.com) and training logs for RefCOCO, RefCOCO+, RefCOCOg. 



<table>
<tr><th> RefCOCO[models](https://www.abc.com) </th><th> RefCOCO+[models](https://www.abc.com) </th><th> RefCOCOg[models](https://www.abc.com) </th></tr>
<tr><td>

| val               | test A            | test B            |
| ----------------- | ----------------- | ----------------- |
| 69.10\%/53.00\% | 74.17\%/57.00\% | 59.75\%/46.96\% |
</td><td>

| val  | test A | test B |
| ---- | ------ | ------ |
| 49.04\% | 51.94\% | 44.31\% |
  
</td><td>
  
| val  | test |
| ---- | ------ |
| 49.04\% | 51.94\% |

</td></tr> </table>


## Visualization

<div align="center">
  <img src="/visual.png"/>
</div><br/>


## <a name="CitingInstNet"></a>Citing Instance-aware-RES

Consider cite Instance-aware-RES in your publications if it helps your research.

```
@inproceedings{xxxx,
  title={xxxx},
  author={Shijia Huang, Tiancheng Shen, Yanwei Li, Shu Liu, Jiaya Jia, Liwei Wang},
  booktitle={xxxx},
  year={xxxx}
}
```
Consider cite this project in your publications if it helps your research. 
```
@misc{Instance-aware-RES,
    author = {Shijia Huang},
    title = {Instance-aware-RES},
    howpublished = {\url{https://github.com/sega-hsj/Instance-aware-RES}},
    year ={2021}
}
```
