# Miscelaneous Code
In this directory, you will find miscelanous code that we used throughout the project. Mainly:

1. How to determine the "concept" of an image (section 4.2 in the paper)
2. Image evaluation metrics (mislabeling, AAR, BAR)


> [!NOTE]
> We tested our code on a Linux machine running Ubuntu 22.04.5 LTS, using Python 3.10.13, with pyenv virtualenv. We recommend using pyenv and Python 3.10.13, as different versions may cause things to break. The steps below will assume you have pyenv and Python 3.10.13 already installed.

## Setup and Installation
A single environment is sufficient for all code in this subdirectory. Note that you could build this off of existing environments (i.e. those created for adversarial mislabeling or fine_tuning).

1. `pyenv virtualenv 3.10.13 amp-sd21-3.10.13`
2. `pyenv activate amp-misc-3.10.13`
3. `pip3 install -r requirements.txt`
4. `python3 -m spacy download en_core_web_sm`

> [!NOTE]
> There are many different CLIP models, we use EVA-2 for concept selection due to its high performance. We use MobileCLIP-B for everything else due to its fast evaluation speed.