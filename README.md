# CP-VTON+

### Original Author

```
@InProceedings{Minar_CPP_2020_CVPR_Workshops,
	title={CP-VTON+: Clothing Shape and Texture Preserving Image-Based Virtual Try-On},
	author={Minar, Matiur Rahman and Thai Thanh Tuan and Ahn, Heejune and Rosin, Paul and Lai, Yu-Kun},
	booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
	month = {June},
	year = {2020}
}
```

<br/>Project [page](https://minar09.github.io/cpvtonplus/)
<br/>Saved/Pre-trained models: [Checkpoints](https://1drv.ms/u/s!Ai8t8GAHdzVUiQA-o3C7cnrfGN6O?e=EaRiFP)
<br/>Dataset: [VITON_PLUS](https://1drv.ms/u/s!Ai8t8GAHdzVUiQQYX0azYhqIDPP6?e=4cpFTI)

## what's new [Oct 2023]

- Run `app.py` for testing or training. 
- It can automatically run both commands (GMM and TOM) and take care of copying files. 
- For training / testing 
  - `subprocess.call(gmm_train/gmm_test, shell=True)`
  - `subprocess.call(tom_train/tom_test, shell=True)`
- fix all the deprecated warning of torch and resolve all isuses regarding dependency.
- have a dedicated branch for only-cpu version.


## Installation and Run

1) create and virtual env.
2) if you are running on cpu, then follow this branch. [CPU](https://github.com/ARajgor/cp-vton-plus/tree/cpu)
3) if you have cuda then install torch with cuda. refer [torch](https://pytorch.org/get-started/locally/)

after that, install the dependencies.
```bash
pip install -r requirements.txt
```

### AutoRun

Run `python app.py`

for tensorboard Run `tensorboard --logdir tensorboard`

### Training

https://github.com/minar09/cp-vton-plus#training

### Testing

https://github.com/minar09/cp-vton-plus#testing

