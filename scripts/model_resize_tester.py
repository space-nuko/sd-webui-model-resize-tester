import os
import os.path
import re
import sys
import gradio as gr
from modules import ui, scripts, script_callbacks, ui_extra_networks, extra_networks, shared, sd_models, sd_vae, sd_samplers, processing, extensions
from scripts import resize_lora


lora = None
xy_grid = None
output_dir = os.path.join(shared.cmd_opts.lora_dir, "model_resize_tester", "resized")
PRECISION_CHOICES = ["float", "fp16", "bf16"]
DYNAMIC_METHOD_CHOICES = ["None", "sv_ratio", "sv_fro", "sv_cumulative"]


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


def apply_dynamic_method(p, x, xs):
    update_script_args(p, True, 0) # set Enabled to True
    update_script_args(p, x, 6)    # enabled, model_type, model_name, model_weight, resize_dim, resize_precision, {dynamic_method}, dynamic_param


def apply_dynamic_param(p, x, xs):
    update_script_args(p, True, 0) # set Enabled to True
    update_script_args(p, x, 7)    # enabled, model_type, model_name, model_weight, resize_dim, resize_precision, dynamic_method, {dynamic_param}


def confirm_models(p, xs):
    for x in xs:
        if x not in lora.available_loras:
            raise RuntimeError(f"Unknown LoRA model: {x}")


def confirm_precision(p, xs):
    for x in xs:
        if x not in PRECISION_CHOICES:
            raise RuntimeError(f"Unknown resize precision: {x}")


def confirm_dynamic_method(p, xs):
    for x in xs:
        if x not in DYNAMIC_METHOD_CHOICES:
            raise RuntimeError(f"Unknown dynamic method: {x}")


def get_loras():
    global lora
    if not lora:
        return ["None"]
    return [x for x in sorted(lora.available_loras)]


def list_available_loras():
    global lora
    if lora:
        lora.list_available_loras()


class Script(scripts.Script):
    def title(self):
        return "Model Resizer"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        load_global_modules()

        is_enabled = False
        with gr.Group():
            with gr.Accordion(self.title(), open=False):
                with gr.Row():
                    enabled = gr.Checkbox(is_enabled, label="Enabled")
                with gr.Row(visible=is_enabled) as row1:
                    model_type = gr.Dropdown(label="Model Type", choices=["LoRA"], value="LoRA")
                    model_name = gr.Dropdown(label="Model Name", choices=get_loras())
                    ui.create_refresh_button(model_name, list_available_loras, lambda: {"choices": get_loras()}, "refresh_model_resize_model_names")
                    model_weight = gr.Slider(minimum=-1.0, maximum=2.0, step=0.05, label="Model Weight", value=1.0)
                with gr.Row(visible=is_enabled) as row2:
                    resize_dim = gr.Slider(minimum=1, maximum=256, step=1, label="Resize Dimension", value=32)
                    resize_precision = gr.Dropdown(label="Resize Precision", choices=PRECISION_CHOICES, value="float")
                with gr.Row(visible=is_enabled) as row3:
                    dynamic_method = gr.Dropdown(label="Dynamic Method", choices=DYNAMIC_METHOD_CHOICES, value="None")
                    dynamic_param = gr.Slider(label="Dynamic Param", minimum=0.0, maximum=256.0, step=0.5)

        def update_visibility(is_enabled):
            return [gr.update(visible=is_enabled)] * 3

        enabled.change(fn=update_visibility, inputs=[enabled], outputs=[row1, row2, row3])

        return [enabled, model_type, model_name, model_weight, resize_dim, resize_precision, dynamic_method, dynamic_param]

    def before_process_batch(self, p, enabled, model_type, model_name, model_weight, resize_dim, resize_precision, dynamic_method, dynamic_param, prompts, *args, **kwargs):
        if not enabled:
            return

        if model_name not in lora.available_loras:
            raise RuntimeError(f"LoRA not found: {model_name}")

        model_type = model_type.lower()
        input_file = lora.available_loras[model_name].filename
        new_model_name = f"{model_name}_{resize_precision}_dim{resize_dim}"
        output_file = os.path.join(output_dir, model_name, new_model_name + ".safetensors")

        if dynamic_method == "None":
            dynamic_method = None

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        if os.path.isfile(output_file):
            print(f"[ModelResizeTester] Using cached model: {new_model_name}.safetensors")
        else:
            print(f"[ModelResizeTester] Resizing model to: {new_model_name}.safetensors")
            resize_lora.resize(resize_precision, resize_dim, input_file, output_file, shared.device, dynamic_method, dynamic_param, verbose=True)

        lora.available_loras[new_model_name] = lora.LoraOnDisk(new_model_name, output_file)

        for i in range(len(prompts)):
            prompts[i] += f" <{model_type}:{new_model_name}:{model_weight}>"

xy_grid = None
lora = None
def load_global_modules():
    global xy_grid, lora

    for scriptDataTuple in scripts.scripts_data:
        if os.path.basename(scriptDataTuple.path) == "xy_grid.py" or os.path.basename(scriptDataTuple.path) == "xyz_grid.py":
            xy_grid = scriptDataTuple.module
            model = xy_grid.AxisOption("[ModelResizer] Model", str, apply_model, xy_grid.format_value_add_label, confirm_models, cost=0.5, choices=get_axis_model_choices)
            model_weight = xy_grid.AxisOption("[ModelResizer] Model Weight", float, apply_model_weight, xy_grid.format_value_add_label, None, cost=0.5, choices=lambda: [0.2, 0.4, 0.6, 0.8, 1])
            dim = xy_grid.AxisOption("[ModelResizer] Dimension", int, apply_dim, xy_grid.format_value_add_label, None, cost=0.5)
            precision = xy_grid.AxisOption("[ModelResizer] Precision", str, apply_precision, xy_grid.format_value_add_label, confirm_precision, cost=0.5, choices=lambda: PRECISION_CHOICES)
            dynamic_method = xy_grid.AxisOption("[ModelResizer] Dynamic Method", str, apply_dynamic_method, xy_grid.format_value_add_label, confirm_dynamic_method, cost=0.5, choices=lambda: DYNAMIC_METHOD_CHOICES)
            dynamic_param = xy_grid.AxisOption("[ModelResizer] Dynamic Param", float, apply_dynamic_param, xy_grid.format_value_add_label, None, cost=0.5)
            xy_grid.axis_options.extend([model, model_weight, dim, precision])
            break

    for s, module in sys.modules.items():
        if module and module.__name__ == "lora" and hasattr(module, "list_available_loras"):
            lora = module
            break

    if not lora:
        print("[ModelResizer] LoRA built-in extension was not loaded, is it enabled in the settings?")
