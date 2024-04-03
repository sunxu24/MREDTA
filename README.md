# MREDTA

A brief description of what this project does and who it's for.

## Getting Started

First, install CUDA 10.2 and CUDNN 8.2.0.
Second, install Anaconda3. Please refer to https://www.anaconda.com/distribution/ to install Anaconda3.
Third, install PyCharm. Please refer to https://www.jetbrains.com/pycharm/download/#section=windows.
Fourth, open Anaconda Prompt to create a virtual environment by the following command:
	conda env create -n env_name -f environment.yml
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites
Code dependencies:
	python '3.7.4' (conda install python==3.7.4)
	pytorch-GPU '1.10.1' (conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch)
	numpy '1.16.5' (conda install numpy==1.16.5)


Note: the environment.yml file should be downloaded and put into the default path of Anaconda Prompt.
What things you need to install the software and how to install them.


## Usage

First, put folder data_kiba, DataHelper.py, emetrics.py, bert_result.py and MREDTA.py into the same folder.
Second, run python bert_result.py in the environment to get features of drug and protein.
Third, modify codes in Demo.py to set the path for loading data and the path for saving the trained model. The details are as follows:
  line 249 in Demo.py
  line 268 in Demo.py
Fourth, open Anaconda Prompt and enter the following command:
  activate env_name
Fifth, run Demo.py in PyCharm.

Expected output：
  The kiba scores between drugs and targets in test set of the small dataset would be output as a csv file.

Expected run time on a "normal" desktop computer:
  The run time in our coumputer (CPU:Xeon 3106, GPU NVIDIA Geforce RTX 2080 Ti, ARM 64G) is about 5 minutes.

Note: in the csv file, drug SMILES, protein sequences and binding affinity values are displayed in column 1, column 2 and column 3, respectively. 


## Contributing

Please read [CONTRIBUTING.md](https://example.com) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://example.com).

## Authors

* **Your Name** - *Initial work* - [Username](https://example.com)

See also the list of [contributors](https://example.com) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Hat tip to anyone whose code was used
- Inspiration
- etc
