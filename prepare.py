import os
import gdown
import zipfile

# Download and prepare dataset for training and testing

data_url = "https://drive.google.com/file/d/1OfFzD-qeXH3Z058K7pQV-nyS4FJT6iA8/view?usp=sharing"
data_file = "viton_plus.zip"
gdown.download(data_url, data_file, fuzzy=True)

with zipfile.ZipFile(data_file, 'r') as zip_ref:
    zip_ref.extractall('data/')

os.remove(data_file)

# Download and prepare checkpoints for GMM and TOM models

ckp_url = "https://drive.google.com/file/d/1xC0f4G2NRg5UILe7XcdqDQtnd12FotRc/view?usp=sharing"
ckp_file = "CP-VTON+.zip"
gdown.download(ckp_url, ckp_file, fuzzy=True)

with zipfile.ZipFile(ckp_file, 'r') as zip_ref:
    zip_ref.extractall('./') # it already has checkpoints folder inside

os.remove(ckp_file)