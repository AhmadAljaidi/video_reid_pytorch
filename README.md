# Recurrent Convolutional Network for Video-based Person Re-Identification

This is pytorch implementation for human Reid described in the paper: [Recurrent Convolutional Network for Video-based Person Re-Identification](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/McLaughlin_Recurrent_Convolutional_Network_CVPR_2016_paper.pdf)

### Prerequisites
1. Pytorch Version --v0.4 with CUDA > 8.0
2. Numpy --v1.14
3. OpenCV --v3.2
4. Matplotlib --v2.1

### Preparing training and testing data
First we need to split the data into train and test

1. Download the [iLIDS-VID](http://www.eecs.qmul.ac.uk/~xiatian/downloads_qmul_iLIDS-VID_ReID_dataset.html) dataset.
2. Run the following command:
``` bash
python prepare_data.py  --dataset_dir=/path/to/i-LIDS-VID/sequences --data_name=<dataset_name>
```

**Note:**  In order to see other changeable parameters such as gen_opt_flow, train_test_split, and frames_per_step run the following command:
``` bash
python prepare_data.py --h
```

### Training
Once the data is successfully prepared, the model can be trained by running the following command:
```bash
python train.py --dataset_dir=/path/to/i-LIDS-VID/sequences --dataset_name=<dataset_name>.txt --checkpoint_dir=/where/to/store/checkpoints
```

**Note:**  In order to see other changeable parameters such as batch size, image height/width, sequence length, etc., run the following command:
``` bash
python train.py --h
```

In order to see the training loss graph open a  `tensorboard` session by
```bash
tensorboard --logdir=./runs/<log_folder> --port 8080
```

### Inference
Once model is trained, we can compute cmc by running the following command:
```bash
python rankCMC_test.py --dataset_dir=/path/to/i-LIDS-VID/sequences --checkpoint_dir=/where/checkpoints/stored --checkpoint_file=hnRiD_latest --n_steps=<number of steps>
```

**Note:**  In order to see other changeable parameters such as image height/width, use_data_aug, etc, run the following command:
``` bash
python rankCMC_test.py  --h
```

### Code citation
Original Code https://github.com/niallmcl/Recurrent-Convolutional-Video-ReID

### Paper citation
```
@inproceedings{mclaughlin2016recurrent,
  title={Recurrent convolutional network for video-based person re-identification},
  author={McLaughlin, Niall and del Rincon, Jesus Martinez and Miller, Paul},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2016 IEEE Conference on},
  pages={1325--1334},
  year={2016},
  organization={IEEE}
}
```
