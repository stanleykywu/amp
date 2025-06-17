# Fine-tuning Text-to-Image Models
In this directory, you will find code we used to fine-tune three different Text-to-Image models (SD21, SDXL, and FLUX). 

> [!NOTE]
> We use [HuggingFace's training script for SD21](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py), but opt for [Kohya's SD-Scripts](https://github.com/kohya-ss/sd-scripts) when dealing w/ SDXL and FLUX as it has better support for large scale training.

> [!CAUTION]
> We **HIGHLY** recommend future users to use the most up-to-date Kohya's SD-Scripts package linked above for fine-tuning. The scripts we include here are included mainly for reference and completeness, but are likely to break as Python packages are updated in the future and will not be maintained. The following scripts are meant to serve primarily as a starting point, especially on how to load arbitrary poison data (see `make_subsets_poison` function in `sdxl_train.py` and `flux_train.py`). 

> [!WARNING]
> We do not include full training data in this repository, as they would take up too much space (over 500k 1024px images). We mainly used [photo-concept-bucket](https://huggingface.co/datasets/bghira/photo-concept-bucket) for our experiments. This repository only includes a small working example of how we did model fine-tuning. 

## Setup and Installation
Each text-to-image model requires a different Python environment. Below, we detail the installation procedure for one such environment and diffusion model. We recommend repeating the process for each diffusion as necessary.

> [!NOTE]
> We tested our code on a Linux machine running Ubuntu 22.04.5 LTS, using Python 3.10.13, with pyenv virtualenv. We recommend using pyenv and Python 3.10.13, as different versions may cause things to break. The steps below will assume you have pyenv and Python 3.10.13 already installed.

1. cd into model directory

        cd sd21

2. Create and activate virtualenv

        pyenv virtualenv 3.10.13 amp-sd21-3.10.13
        pyenv activate amp-sd21-3.10.13

3. Install required packages

        pip3 install -r requirements.txt

## Usage
For each type of diffusion model, we use HuggingFace's `accelerate` library to distribute training across multiple GPUs (we used 4 per model). Each diffusion model includes a shell script, which takes in 4 arguments:

1. Comma-separated string of the available CUDA gpus (e.g. "0,1,2,3")
2. Integer representing the number of processes (equal to number of gpus)
3. accelerate library's main process port (e.g. 25000, 25001). Note multiple training scripts cannot occupy the same process port
4. String filepath to a json config, examples of which can be found in `./configs` dir within each model subdirectory.

As noted above, we do not include our full fine-tuning dataset due to size, but we do include a minimal dataset in this repository to show how training can be done. A small set of 100 images is included in this subdirectory under [`./example_clean_data`](./example_clean_data), and an example set of 5 poison images (they are not actual poison) images can be found under [`./example_poison_data`](./example_poision_data)

### Setting up Dataset Directory
For training, our scripts require the following (note that we also provide examples in `./example_clean_data` and `./example_poison_data`). The same general structure is needed for both clean data and poison data:

1. A directory containing `.png` files, the path for which should be provided in `clean_data_dir` and `poison_data_dir` respectively. Note that more than one `poison_data_dir` may be provided to load different kinds of poison image data.
2. A `metadata.jsonl` file following [HuggingFace's image dataset structure](https://huggingface.co/docs/datasets/v2.7.1/en/image_dataset), containing each image's caption for each image directory.
3. A third `json` file that specifies which images in the directory to select for clean training, including resolution and captions. The number of entries in this json file must match the `total_data` parameter in the training config. This is only used for the clean dataset, since it is more convenient to have all images in one directory location. We assume all poison data will be used and separated them out in separate directories.

The `metadata.jsonl` file will be automatically detected by training scripts, and does not need to be specified anywhere. However, the third `json` file needs to be manually specified as the `training_images_to_select` parameter in your training config (see provided test configs).

### Fine-tuning SD2.1
We use the publicly available [stable-diffusion-2-1](stabilityai/stable-diffusion-2-1) for training. Since no modifications to this model are needed, our provided scripts will pull the model straight from HuggingFace (no changes to `pretrained_model_name_or_path` parameter is needed).

```
pyenv activate amp-sd21-3.10.13
cd sd21-ft
sh sd21-train.sh "0,1,2,3" 4 25000 {path_to_json_config}
```

### Fine-tuning SDXL
We used [stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) in FP16 with a more stable [VAE](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) designed to better operate in FP16. You can load and save the model locally via the following snippet:

```
import torch
from diffusers import DiffusionPipeline, AutoencoderKL

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
pipe.save_pretrained("./path/to/output_dir")
```

Once saved, modify the json configuration file to point `pretrained_model_name_or_path` to the location of the saved SDXL model.

```
pyenv activate {sdxl_environment}
cd sdxl-ft
sh sdxl-train.sh "0,1,2,3" 4 25001 {path_to_json_config}
```

### Fine-tuning FLUX
For FLUX, you will first need to obtain access to the [FLUX HuggingFace repo](https://huggingface.co/black-forest-labs/FLUX.1-dev). Then, you will need to download 4 safetensor files:

1. [ae.safetensors](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors)
2. [flux1-dev.safetensors](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/flux1-dev.safetensors)
3. [clip_l.safetensors](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/clip_l.safetensors)
4. [t5xxl_fp16.safetensors](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors)

Once downloaded, please set the corresponding entries in your json config to point to the location of said files.

```
pyenv activate {flux_environment}
cd flux-ft
sh flux-train.sh "0,1,2,3" 4 25002 {path_to_json_config}
```
