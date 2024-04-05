# MREDTA

In this work, we build a new neural network based on Transformer Encoder and BERT to predicting DTA score. We train the model on KIBA and Davis Dataset and the model shows state-of-the-art performance compared to 10 existing neural network methodologies.

## Requirements
Code dependencies:

	python '3.7.4' (conda install python==3.7.4)
 
	pytorch-GPU '1.10.1' (conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch)
 
	numpy '1.16.5' (conda install numpy==1.16.5)


## Files
`bert_result.py` is the feature extractor we use to convert drug and protein sequences into features extracted by BERT.

`MREDTA.py` contains the model and the main function for training and testing.

`emetrics.py` is the calculating tool for MSE, CI, and $r_m^2$.

`DataHelper.py` helps to convert drug and protein sequences into digital sequences through drug and protein vocabularies.

`Network_test.py` provides DTA scores between drugs and targets in test set as a csv file.


## BERT Model

Here is the pre-trained BERT model we use for drugs. The BERT model should be put in the folder `pre-trained_BERTmodel`.

[model](https://huggingface.co/dmis-lab/biobert-v1.1)  


