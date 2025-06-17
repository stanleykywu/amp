# Generating Adversarial Images 
In this directory, you will find code that we used to generate ***white-box targeted*** adversarial perturbations for the three VLMS: CogVLM, BLIP-3 (xgen-MM), and LLaVA-1.5 (7b variant). This attack was used for the main result in our paper (section 5).

> [!WARNING]
> Although each VLM is available on HuggingFace, there are cross-dependencies (transformer version) that are not compatible for each VLM to be used in a single Python environment. Thus, we include separate `requirements.txt` for each VLM, under their own separate directory.

## Setup & Installation
As mentioned above, each VLM requires a different Python environment. Below, we detail the installation procedure for one such environment and VLM. We recommend repeating the process for each VLM as necessary.

> [!NOTE]
> We tested our code on a Linux machine running Ubuntu 22.04.5 LTS, using Python 3.10.13, with pyenv virtualenv. We recommend using pyenv and Python 3.10.13, as different versions may cause things to break. The steps below will assume you have pyenv and Python 3.10.13 already installed.

1. cd into model directory

        cd cogvlm

2. Create and activate virtualenv

        pyenv virtualenv 3.10.13 amp-cogvlm-3.10.13
        pyenv activate amp-cogvlm-3.10.13

3. Install required packages

        pip3 install -r requirements.txt

## Usage
Each script generates adversarial perturbation against a single model, using both an input image, and a target image (will add perturbation to the input image to cause it to mislabel as the target image). The arguments for all three attack scripts (`attack_cogvlm.py`, `attack_xgenmm.py`, `attack_llava.py`) are the same. Below are three examples, one for each model. We provide 2 example images to use for quick testing.

Attacking CogVLM (~5.5 minutes on a single A100 GPU)
```
pyenv activate amp-cogvlm-3.10.13
CUDA_VISIBLE_DEVICES=0 python3 cogvlm/attack_cogvlm.py --input_image_fp=./source_img.png --target_image_fp=./target_img.png --output_image_fp=./cogvlm/amp-cogvlm.png --verbose
```

Attacking XGen-MM (~5 minutes on a single A100 GPU)
```
pyenv activate {xgenmm_environment}
CUDA_VISIBLE_DEVICES=1 python3 xgen_mm/attack_xgenmm.py --input_image_fp=./source_img.png --target_image_fp=./target_img.png --output_image_fp=./xgenmm/amp-xgenmm.png --verbose
```

Attacking LLaVA (~2 minutes on a single A100 GPU)
```
pyenv activate {llava_environment}
CUDA_VISIBLE_DEVICES=2 python3 llava/attack_llava.py --input_image_fp=./source_img.png --target_image_fp=./target_img.png --output_image_fp=./llava/amp-llava.png --verbose
```

### Testing
We also provide notebooks to test our adversarial attack's mislabeling effect. For the provided two source and target images, the image should look like the source, but get labeled as if it was the target.
