**[News]🎇CoTAM is accepted to the Findings of ACL2024!🎇**

# Chain-of-Thoughts Attribute Manipulation (CoTAM) 
**CoTAM** ([arxiv.org/abs/2307.07099](https://arxiv.org/abs/2307.07099)) is an LLM-based framework that generates efficient training data for smaller language models.
![CoTAM](https://github.com/KomeijiForce/CoTAM/blob/main/cotam-v2.jpg)

## :wrench:Manipulate Your Texts!
(Currently only SST-2 example is available, we will upload a unified version later)

Generate (You have to put your OpenAI API key in constant.py)
```
python cotam.py
```
Fine-tuning
```
python tune.py
```
Nearest Centroid (NC)
```
python nc.py
```
K-Nearest Neighbors (KNN)
```
python knn.py
```

## :bulb:Performance
**Fine-tuning Results**
| **Method** | SST-2 | TweetEmo | AG-NEWS | MNLI(m) | MNLI(mm) | MRPC | CSQA |
| --- | --- | --- | --- | --- | --- | --- | --- |
| K-Shot (Human) | 60.54 | 44.38 | 81.05 | 35.88 | 38.75 | 51.96 | 34.54 |
| NK-Shot (Human) | 62.17 | 69.51 | 88.66 | 43.33 | 44.03 | 57.50 | 47.36 |
| NK-Shot (LLM) | 61.14 | 69.11 | 85.64 | 41.71 | 42.92 | 55.88 | 45.12 |
| K-FlipDA++ | 74.28 | 70.87 | 84.72 | 51.52 | 53.56 | 60.15 | 50.52 |
| K-CoTDA | 70.83 | 67.76 | 85.19 | 36.06 | 36.28 | 55.54 | 48.79 |
| K-CoTAM | **79.12** | **72.76** | **85.80** | **54.07** | **56.16** | **61.64** | **53.22** |

**Instance-based Text Classification Results**
| **Method** | SST-2 (NC) | SST-2 (KNN) | TweetEmo (NC) | TweetEmo (KNN) | AG-NEWS (NC) | AG-NEWS (KNN) |
| --- | --- | --- | --- | --- | --- | --- |
| K-Shot (Human) | 82.00 | 78.20 | 66.01 | 59.92 | 77.72 | 73.57 |
| NK-Shot (Human) | 87.55 | 83.45 | 71.23 | 67.56 | 84.70 | 82.33 |
| NK-Shot (LLM) | 86.78 | 80.26 | 69.34 | 64.90 | **81.19** | **79.34** |
| K-FlipDA++ | 88.13 | 86.76 | 66.53 | 65.05 | 79.82 | 75.11 |
| K-CoTDA | 86.38 | 83.00 | 68.63 | 61.58 | 78.87 | 76.56 |
| K-CoTAM | **88.43** | **87.52** | **70.02** | **65.37** | 80.60 | 75.48 |

## :mag_right:Further Exploration

### Principal Component Analysis
![PCA](https://github.com/KomeijiForce/CoTAM/blob/main/pca.png)

### Data Scale Analysis
![PCA](https://github.com/KomeijiForce/CoTAM/blob/main/data_scale.png)

## :paw_prints:Citation
```
@article{peng2023cotam,
  title={Generating Efficient Training Data via LLM-based Attribute Manipulation},
  author={Peng, Letian and Zhang, Yuwei and Shang, Jingbo},
  journal={arXiv preprint arXiv:2307.07099},
  year={2023}
}
```
