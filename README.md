# Stable Diffusion Save intermediate images extension 

A custom extension for [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) that implements saving intermediate images.

<img src="images/extension.png"/>

## Installation

The extension can be installed directly from within the **Extensions** tab within the Webui.

You can also install it manually by running the following command from within the webui directory:

	git clone https://github.com/AlUlkesh/sd_save_intermediate_images/ extensions/sd_save_intermediate_images

## Output

Once the image generation begins, the intermediate images will start saving in a directory under a new \outputs\txt2img-images\intermediates directory.

## Limitations
Does not work with DDIM and PLMS
