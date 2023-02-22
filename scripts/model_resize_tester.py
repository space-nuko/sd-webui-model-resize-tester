import os
import os.path
import re
import sys
import gradio as gr
from modules import ui, scripts, script_callbacks, ui_extra_networks, extra_networks, shared, sd_models, sd_vae, sd_samplers, processing
from scripts import resize_lora


lora = None
xy_grid = None
output_dir = os.path.join(shared.cmd_opts.lora_dir, "model_resize_tester", "resized")
PRECISION_CHOICES = ["float", "fp16", "bf16"]


def get_axis_model_choices():
    return list(lora.available_loras.keys())


def update_script_args(p, value, arg_idx):
    if isinstance(p, processing.StableDiffusionProcessingTxt2Img):
        all_scripts = scripts.scripts_img2img
    elif isinstance(p, processing.StableDiffusionProcessingImg2Img):
        all_scripts = scripts.scripts_img2img
    else:
        raise Exception(f"Unknown processing: {p}")

    for s in all_scripts.alwayson_scripts:
        if isinstance(s, Script):
            args = list(p.script_args)
            print(f"Args: {args}")
            print(f"Changed arg {arg_idx} from {args[s.args_from + arg_idx]} to {value}")
            args[s.args_from + arg_idx] = value
            p.script_args = tuple(args)
            break


def apply_model(p, x, xs):
    update_script_args(p, True, 0) # set Enabled to True
    update_script_args(p, x, 2)    # enabled, model_type, {model_name}, model_weight, resize_dim, resize_precision


def apply_model_weight(p, x, xs):
    update_script_args(p, True, 0) # set Enabled to True
    update_script_args(p, x, 3)    # enabled, model_type, model_name, {model_weight}, resize_dim, resize_precision


def apply_dim(p, x, xs):
    update_script_args(p, True, 0) # set Enabled to True
    update_script_args(p, x, 4)    # enabled, model_type, model_name, model_weight, {resize_dim}, resize_precision


def apply_precision(p, x, xs):
    update_script_args(p, True, 0) # set Enabled to True
    update_script_args(p, x, 5)    # enabled, model_type, model_name, model_weight, resize_dim, {resize_precision}


def confirm_models(p, xs):
    for x in xs:
        if x not in lora.available_loras:
            raise RuntimeError(f"Unknown LoRA model: {x}")


def confirm_precision(p, xs):
    for x in xs:
        if x not in PRECISION_CHOICES:
            raise RuntimeError(f"Unknown resize precision: {x}")


class Script(scripts.Script):
    def title(self):
        return "Model Resizer"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        global lora
        for s, module in sys.modules.items():
            # print(s)
            # print(module.__name__)
            # print(dir(module))
            if module.__name__ == "lora" and hasattr(module, "list_available_loras"):
                lora = module
                break
        assert lora

        with gr.Group():
            with gr.Accordion(self.title(), open=False):
                with gr.Row():
                    enabled = gr.Checkbox(False, label="Enabled")
                with gr.Row():
                    model_type = gr.Dropdown(label="Model Type", choices=["LoRA"], value="LoRA")
                    model_name = gr.Dropdown(label="Model Name", choices=[x for x in lora.available_loras])
                    model_weight = gr.Slider(minimum=-1.0, maximum=2.0, step=0.05, label="Model Weight", value=1.0)
                    ui.create_refresh_button(model_name, lora.list_available_loras, lambda: {"choices": [x for x in lora.available_loras]}, "refresh_model_resize_model_names")
                with gr.Row():
                    resize_dim = gr.Slider(minimum=1, maximum=256, step=1, label="Resize Dimension", value=32)
                    resize_precision = gr.Dropdown(label="Resize Precision", choices=PRECISION_CHOICES, value="float")

        return [enabled, model_type, model_name, model_weight, resize_dim, resize_precision]

    def before_process_batch(self, p, enabled, model_type, model_name, model_weight, resize_dim, resize_precision, prompts, **kwargs):
        if not enabled:
            return

        if model_name not in lora.available_loras:
            raise RuntimeError(f"LoRA not found: {model_name}")

        model_type = model_type.lower()
        input_file = lora.available_loras[model_name].filename
        new_model_name = f"{model_name}_dim{resize_dim}_{resize_precision}"
        output_file = os.path.join(output_dir, model_name, new_model_name + ".safetensors")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        if os.path.isfile(output_file):
            print(f"[ModelResizeTester] Using cached model: {new_model_name}.safetensors")
        else:
            print(f"[ModelResizeTester] Resizing model to: {new_model_name}.safetensors")
            resize_lora.resize(resize_precision, resize_dim, input_file, output_file, shared.device, verbose=True)
            lora.available_loras[new_model_name] = lora.LoraOnDisk(new_model_name, output_file)

        for i in range(len(prompts)):
            prompts[i] += f" <{model_type}:{new_model_name}:{model_weight}>"


xy_grid = None
for scriptDataTuple in scripts.scripts_data:
    if os.path.basename(scriptDataTuple.path) == "xy_grid.py" or os.path.basename(scriptDataTuple.path) == "xyz_grid.py":
        xy_grid = scriptDataTuple.module
        model = xy_grid.AxisOption("Resizer Model", str, apply_model, xy_grid.format_value_add_label, confirm_models, cost=0.5, choices=get_axis_model_choices)
        model_weight = xy_grid.AxisOption("Resizer Model Weight", float, apply_model_weight, xy_grid.format_value_add_label, None, cost=0.5, choices=lambda: [0.2, 0.4, 0.6, 0.8, 1])
        dim = xy_grid.AxisOption("Resizer Dimension", int, apply_dim, xy_grid.format_value_add_label, None, cost=0.5)
        precision = xy_grid.AxisOption("Resizer Precision", str, apply_precision, xy_grid.format_value_add_label, confirm_precision, cost=0.5, choices=lambda: PRECISION_CHOICES)
        xy_grid.axis_options.extend([model, model_weight, dim, precision])
