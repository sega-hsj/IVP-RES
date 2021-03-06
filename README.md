# IVP-RES

**Referring Expression Segmentation with Instance-aware Visual Priors**

<!-- [[`arXiv`](https://arxiv.org/pdf/xxxx.xxxx.pdf)] [[`BibTeX`](#CitingIVP)] -->

<div align="center">
  <img src="/framework.png"/>
</div><br/>

## Setup

We recommended the following dependencies.

* Python 3.6
* Numpy

<!-- This code is derived from [RRN](https://github.com/liruiyu/referseg_rrn) \[2\]. Please refer to it for more details of setup. -->

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

<!-- * Glove Embedding -->

<!-- Download `Gref_emb.npy` and `referit_emb.npy` and put them in `data/`. We provide download link for Glove Embedding here: -->
<!-- [Baidu Drive](https://pan.baidu.com/s/19f8CxT3lc_UyjCIIE_74FA), password: 2m28. -->


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

## Pre-trained Models and Checkpoints

Pending...

<!-- <table>
<tr><th> RefCOCO</th><th> RefCOCO+ </th><th> RefCOCOg </th></tr>
<tr><td>

| val               | test A            | test B            |
| ----------------- | ----------------- | ---------- |
| 67.40\% | 70.65\% | 64.32\% |
</td><td>

| val  | test A | test B |
| ---- | ------ | ------ |
| 57.32\% | 62.57\% | 49.15\% |
  
</td><td>
  
| val  | test |
| ---- | ------ |
| 56.05\% | 56.83\% |

</td></tr> </table>
 -->

## Visualization

Get visualization results with:
```
python myvis.py --config-file configs/my_Model_soft2-unc-3x.yaml --num-gpus 1 MODEL.WEIGHTS ./output/my_Model_soft2_ab2/unc/1x/model_final.pth DATASETS.TEST \(\"unc_val\",\)
```

<div align="center">
  <img src="/visual.png"/>
</div><br/>


## <a name="CitingInstNet"></a>Citing IVP

Pending...
<!-- Consider cite Instance-aware-RES in your publications if it helps your research. -->
