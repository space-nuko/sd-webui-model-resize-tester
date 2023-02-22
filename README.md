# stable-diffusion-webui-model-resize-tester

This extension for [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) lets you batch resize a LoRA with different dimensions and test the differences between the output LoRAs on the fly.

## Usage

Under the `Model Resizer` tab in the `txt2img` interface, select a LoRA and the dimensions/precision to resize it with. `Model Weight` will control the weight of the resized LoRA in the generated image. When you generate an image, a new LoRA is created via the resizing script if one has not been created yet for those parameters and it will activated for the batch. You don't need to add the `<lora:lora_name:1>` syntax to the prompt yourself, it will be added by the extension automatically.

You can use this extension with the `XYZ Grid` script to test different resizing settings in several batches. Use the `Resizer Model`, `Resizer Model Weight`, `Resizer Dimensiontts` and `Resizer Precision` options to control the extension.

The LoRAs that are created by the resizing script will be saved to your `lora_dir`, usually `models/lora`, under the `model_resize_tester` subdirectory.
