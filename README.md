# MREDTA
In this work, we developed a promising deep learning model to predict DTA based on BERT and Transformer (MREDTA). We evaluate the performance of the MREDTA on KIBA and Davis datasets, and the results show that MREDTA have superior performance compared to ten advanced models.

## Requirements
Code dependencies:

	python '3.7.4' (conda install python==3.7.4)
 
	pytorch-GPU '1.10.1' (conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch)
 
	numpy '1.16.5' (conda install numpy==1.16.5)


## Files
`bert_result.py` is the feature extractor we use to convert drug SMILES and protein sequences into features extracted by BERT.

`MREDTA.py` contains the model and the main function for training and testing.

`emetrics.py` is the calculating tool for MSE, CI, and $r_m^2$.

`DataHelper.py` helps to convert drug SMILES and protein sequences into digital sequences through drug and protein vocabularies.

`Network_test.py` provides DTA scores between drugs and targets in test set as a csv file.


## BERT Model

Here is the pre-trained BERT model we use for drugs. The BERT model should be put in the folder `pre-trained_BERTmodel` which should be placed in the same directory as the other code files.

[model](https://huggingface.co/dmis-lab/biobert-v1.1)  


