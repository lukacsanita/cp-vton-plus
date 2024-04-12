# Reimplementation of CP-VTON+

## Original author

Project [page](https://minar09.github.io/cpvtonplus/)

## Overview

**CP-VTON+** is an advanced method for virtual try-on, allowing users to visualize how clothing items would look on them without physically wearing them. It combines clothing shape and texture preservation to create realistic virtual try-on results.

It is a combination of consecutive training and testing of GMM (Geometric Matching Module) and TOM (Try-On Module).
- **GMM** generates warped clothes according to the target human.
- **TOM** blends the warped clothes outputs from **GMM** into the target human properties to generate the final try-on output

## Running with Google Colab

- Import the notebook `colab.ipynb` to **Google Colab**
- Change the Colab Runtime Type to **GPU**
- Execute all cells in the notebook to run the scripts for:
	- Git clone this repo
  	- Prepare model checkpoints and dataset.
  	- Test the GMM and TOM models with test dataset.
  	- Show the test results in `smart_result` directory.


## Running locally

### Installation

1. Ensure you have Python 3 installed.

2. Install the required packages using pip:
	```
	pip install -r requirements.txt
	```

3. Download and prepare the dataset:

	- The full and processed dataset is available [here](https://1drv.ms/u/s!Ai8t8GAHdzVUiQQYX0azYhqIDPP6?e=4cpFTI).
	- After downloading, unzip and move all contents to the `data` directory.

4. Download and prepare the checkpoints:

	- The saved checkpoints is available [here](https://1drv.ms/u/s!Ai8t8GAHdzVUiQA-o3C7cnrfGN6O?e=EaRiFP).
	- After downloading, unzip and move  `data` directory.

### End-to-end run Testing

Run the testing end-to-end with GMM and TOM for the test dataset

```
python e2e.py --mode test
```

View the Tensorboard traning logs:
```
tensorboard --logdir tensorboard
```

### End-to-end run Training

Run the training end-to-end with GMM and TOM for the train dataset

```
python e2e.py --mode train
```

View the Tensorboard testing logs:
```
tensorboard --logdir tensorboard
```
