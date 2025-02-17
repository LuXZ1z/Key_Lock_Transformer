## **[Key_Lock_Transformer](https://github.com/LuXZ1z/Key_Lock_Transformer)**



## Table of Contents

- [Installation](#Installation)
- [Quick Start](#QS)
- [Project Structure](#PS)
- [Training](#T)
- [Inference](#I)
- [Configuration](#C)
- [Acknowledgments](#A)

------

## Installation<a id="Installation"></a>

### Prerequisites

- Python 3.8+
- pip package manager

### Install Dependencies

```
pip install pandas scipy tqdm pickle5 seaborn matplotlib scikit-learn
```

------

## Quick Start<a id="QS"></a>

1. **Prepare Data**

   - Place your dataset in `./input/original_data`.

   - The dataset should be a CSV file with the following format:

   | **seq**                                  | **sum**  |
   | :--------------------------------------- | :------- |
   | 16nt DNA sequence: [Lock(8nt):Key (8nt)] | Activity |

2. **Run Training & Prediction**

   ```
   python main.py
   ```

------

## Project Structure<a id="PS"></a>

```
│  BMC_loss.py                                                                                                  
│  config.py
│  dataloader.py
│  embedding.py
│  environment.yaml
│  infer.py
│  main.py
│  model.py
│  MyTransformer.py
│  readme.md
│  train.py
│  utils.py
│
├─cache
├─figs
├─input
│  │
│  └─original_data			# Place your dataset here      
│
└─pretrained			# Download from https://github.com/pnpnpn/dna2vec
        dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v
        word_indices.csv
```

------

## Training<a id="T"></a>

To train the model:

```
python train.py
```

*Configure hyperparameters in `config.py` (e.g., batch size, learning rate, model architecture).*

------

## Inference<a id="I"></a>

To run predictions:

```
python infer.py
```

------

## Configuration<a id="C"></a>

Key parameters in `config.py` include:

- Dataset paths
- Model architecture settings
- Training hyperparameters (epochs, batch size, learning rate)
- I/O configurations

*Modify these parameters to adapt to your specific use case.*

------

## Acknowledgments<a id="A"></a>

This project leverages code and methodologies from these open-source repositories:

- [Transformer Translation](https://github.com/moon-hotel/TransformerTranslation) - Transformer architecture implementation
- [DNA2Vec](https://github.com/pnpnpn/dna2vec) - Biological sequence embedding framework
- [BMCLoss](https://github.com/jiawei-ren/BalancedMSE) - Balanced MSE for Imbalanced Visual Regression



