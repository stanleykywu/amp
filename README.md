# AMP: Adversarial Mislabeling for Poisoning

This repository contains the main research code for our paper "On the Feasibility of Poisoning Text-to-Image AI Models via Adversarial Mislabeling" - S. Wu, R. Bhaskar, A. Ha, S. Shan, H. Zheng, B. Zhao (accepted to CCS 2025)

## Overview

We include code for two main parts of our research:

1. Generating adversarial images that fool VLMs (`./adversarial_mislabeling_attack`)
    * We include our ***white-box targeted*** attack (section 5 of paper) against all three VLMs we evaluated against (CogVLM, xGen-MM, and LLaVA)
    * Setup and usage can be found [here](./adversarial_mislabeling_attack//README.md)

2. Fine-tuning text-to-image models (`./fine_tuning`)
    * We include our fine-tuning scripts for all three text-to-images we evaluated against (SD21, SDXL, and FLUX)
    * Setup and usage can be found [here](./fine_tuning//README.md)
