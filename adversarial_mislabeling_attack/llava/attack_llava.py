import argparse

import torch
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import LlavaForConditionalGeneration


def fraction_to_float(value):
    try:
        return float(eval(value))
    except (SyntaxError, NameError, ZeroDivisionError) as e:
        raise argparse.ArgumentTypeError(f"Invalid fraction: {value}")


def main(args):
    # model setup
    dtype = torch.float16
    model_id = "llava-hf/llava-1.5-7b-hf"
    llava_model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    vision_tower = llava_model.vision_tower.cuda()

    # torchvision preprocessing
    llava_image_size = 336
    to_tensor = transforms.ToTensor()
    transform = transforms.Compose(
        [
            transforms.Resize(
                (llava_image_size, llava_image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    # source image setup
    source_img = Image.open(args.input_image_fp)
    source_tensor = to_tensor(source_img).to("cuda", dtype)
    modifier = torch.clone(source_tensor) * 0.1

    # target image setup
    target_img = Image.open(args.target_image_fp)
    target_tensor = to_tensor(target_img).to("cuda", dtype)
    target_feature = vision_tower(
        torch.stack([transform(target_tensor)]), output_hidden_states=True
    ).hidden_states[-2]

    # get optimization parameters
    initial_lr = args.initial_lr
    num_optimization_steps = args.num_optimization_steps
    budget = args.budget

    for i in tqdm(
        range(num_optimization_steps),
        desc=f"[llava]: generating perturbation",
        disable=not args.verbose,
    ):
        # linear learning rate scheduling
        alpha = (
            initial_lr - (initial_lr - initial_lr / 100) / num_optimization_steps * i
        )
        modifier.requires_grad_(True)

        # current adversarial image's feature
        adv_tensor = torch.clamp(modifier + source_tensor, 0, 1)
        adv_feature = vision_tower(
            torch.stack([transform(adv_tensor)]), output_hidden_states=True
        ).hidden_states[-2]

        # loss and backprop w/ pgd
        loss = (adv_feature - target_feature).norm()
        grad = torch.autograd.grad(loss, modifier)[0]
        modifier = modifier.detach()
        modifier = modifier - torch.sign(grad) * alpha
        modifier = torch.clamp(modifier, min=-budget, max=budget)

    # get final adversarial image and save
    adv_image = source_tensor + modifier
    adv_image = torch.clamp(adv_image, 0.0, 1.0)
    adv_img = transforms.ToPILImage()(adv_image.to(torch.float16))
    adv_img.save(args.output_image_fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image_fp", type=str, required=True)
    parser.add_argument("--target_image_fp", type=str, required=True)
    parser.add_argument("--output_image_fp", type=str, required=True)
    parser.add_argument(
        "--budget",
        type=fraction_to_float,
        default=(16 / 255),
        help="l_inf perturbation budget to use for adversarial image",
    )
    parser.add_argument(
        "--num_optimization_steps",
        default=4000,
        help="number of iterations to do optimization",
    )
    parser.add_argument(
        "--initial_lr", default=0.005, help="initial learning rate for optimization"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="setting this flag will include a tqdm progress bar",
    )
    args = parser.parse_args()

    main(args)
