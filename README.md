# Lung Sound Recognition Method Based on Multi-Resolution Interlaced Net and Time-Frequency Feature Enhancement


## Introduction

The Code Repository for  "[Lung Sound Recognition Method Based on Multi-Resolution Interlaced Net and Time-Frequency Feature Enhancement]".

Air pollution and an aging population have caused the increasing rates of lung diseases and elderly lung diseases year by year. At the same time, the outbreak of COVID-19 has brought challenges to the medical system. This raises higher requirements for the prevention and diagnosis of lung diseases. Analyzing lung sound signals by using artificial intelligence technology could alleviate the pressure of lung disease diagnosis. Currently, it is difficult for most lung sound recognition models to maintain the correlation between time domain and frequency domain information. And the models based on convolutional neural network hard to focus on multi-scale features at different resolutions. Feature fusion also ignores the differences in the influence between time and frequency features. To address these issues, a lung sound recognition model based on multi-resolution interlaced net and time-frequency feature enhancement is proposed, which consists of a heterogeneous dual-branch time-frequency feature extractor, a time-frequency feature enhancement module based on branch attention, and a fusion semantic classifier based on semantic mapping. The feature extractor independently extracts the frequency domain and time domain information of lung sound signals through a multi-resolution interlaced net and Transformer, which maintains the correlation between time-frequency features. The feature enhancement module focuses on the differences in the influence of frequency domain and time domain information on prediction results by branch attention. The classifier maps the output feature map to a semantic map through semantic downsampling and then performs the classification, which considers both parameter quantity and accuracy. Experimental results on a combined dataset show that the proposed model outperforms other models, with an accuracy improvement of over 2.13%.

![Architecture](fig/arch.png)

## Classification Results on Combinded dataset and ESC-50(%)

<p align="center">
<img src="fig/ac_result.png" align="center" alt="ClS Result" width="100%"/>
</p>


## Getting Started

### Install Requirments
```
pip install -r requirements.txt
```

[PyTorch](https://pytorch.org/) packages are not included in requirement, so you can download them yourself based on your server's configuration. 

Install the 'SOX' and the 'ffmpeg', we recommend that you run this code in Linux inside the Conda environment. In that, you can install them by:
```
sudo apt install sox 
conda install -c conda-forge ffmpeg
```

### Prepare for Datasets
*Combined Dataset.

Because the dataset involves trade secrets, it cannot be made public at this time. If you would like to use it for your own research, you can contact us and we will consider sharing the dataset after evaluation.

*[ESC-50](https://github.com/karolpiczak/ESC-50)

You can download the dataset yourself. 
Then set:
```
dataset_path = "the download address of the dataset"
resample_path = "resampled directory"
```
Run:
```
cd esc-50
python esc50_data.py
```

### Train and Evaluation

All scripts is run by main.py:
```
Train: CUDA_VISIBLE_DEVICES=1,2 python main.py train
Test: CUDA_VISIBLE_DEVICES=1,2 python main.py test
```
