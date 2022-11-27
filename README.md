# Audiance
A platform for **visualizing**, **analyzing** and **generating** music and audio with deep learning.

The name's a portmanteau of a bunch of words like Audio, Audience and Ambiance.

## Requirements
Python 3.8 and Pytorch 1.12.1+cu116

Developed on an RTX 3070, so expect a maximum of 8 GB VRAM needed for all computation. Will note if I spend more on an instance and the VRAM requirements increase.

See requirements.txt for other python libraries used..

## To Do
- [ ] Visualize Music
  - [x] As Waveplots
  - [x] As Spectrograms
  - [ ] As MFCCs
  - [ ] With Embeddings
- [ ] Implement Datasets
  - [x] GTZAN
  - [ ] FMA
  - [ ] MTG
- [x] Predict Music Genres
  - [x] Inference with Pretrained Model
  - [x] Train from Scratch
  - [x] Within existing classes
  - [ ] Zero-Shot
- [ ] Recommend Similar Music
  - [ ] Based on Visual embeddings
  - [ ] Based on Audio embeddings
- [ ] Generate new Music
  - [ ] GAN-Based Model
  - [ ] Diffusion-Based Model
- [ ] Music2Music
  - [ ] Extend from Generation models

## Arguments

| argument | description    | possible values |
|----------|----------------|-----------------|
| -t       | b              | d               |

#### To Train for Classification:
```
Run scripts/train.py
```

#### To Train for Generation:
```
tbd
```

#### To Evaluate for Classification:
```
tbd
```

#### To Evaluate for Generation:
```
tbd
```

## Datasets Used
| Dataset | URL                          | Notes                                                                                                                                                       |
|---------|------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GTZAN   | [Dataset on Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) | Yes, it's not a *great* dataset but it's a classic one. Train/Test/Val Split was done manually but I may add a splitting script later |

## Folder Structure
Somewhat tentative and may change in the future.
```
│
├─── data\
│    ├─── gtzan\
│    └─── other datasets\
│
├─── datasets\
│    ├─── preprocessing\
│    │    ├─── __init__.py
│    │    └─── transforms.py
│    ├─── __init__.py
│    └─── gtzan_dataset.py
│
├─── models\
│    ├─── classify\
│    │    ├─── __init__.py
│    │    ├─── custom_mlp.py
│    │    └─── pretrained_model.py
│    └─── generate\
│         ├─── __init__.py
│         └─── coming_soon.py
│
├─── output\
│    ├─── results\
│    └─── saved_models\
│
├─── trainers\
│    ├─── __init__.py
│    └─── classification_trainer.py
│
├─── scripts\
│    ├─── evaluate.py
│    └─── train.py
│
├─── utils\
│    ├─── plotting.py
│    ├─── preprocess_data.py
│    ├─── process_imgs.py
│    ├─── regenerate_audio.py
│    └─── visualize_audio.py
│
├─── .gitignore
├─── main.py
├─── README.md
└─── requirements.txt
```

### Expected Folder Structure for Datasets
#### GTZAN
```
└─── gtzan\
     ├─── train\
     │    ├─── audio\
     │    │    ├─── [genres]
     │    │    │    ├─── [songs].wav
     │    └─── image\
     │    │    ├─── [genres]
     │    │    │    ├─── [songs].png
     ├─── test\
     │    ├─── <identical to train>
     └─── val\
          ├─── <identical to train>
```