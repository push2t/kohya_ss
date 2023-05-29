import json
import os
import regex
import glob
import argparse
import random
import math
import subprocess
import pathlib

DEBUG=1

class SomethinError(Exception): pass

def run_cmd_sample(
    sample_every_n_steps,
    sample_every_n_epochs,
    sample_sampler,
    sample_prompts,
    output_dir,
):

    # awas note:
    # i'm pretty sure this is a new feature that isnt in my configs.
    if sample_prompts is None:
        return ""
    output_dir = os.path.join(output_dir, 'sample')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    run_cmd = ''

    if sample_every_n_epochs == 0 and sample_every_n_steps == 0:
        return run_cmd

    # Create the prompt file and get its path
    sample_prompts_path = os.path.join(output_dir, 'prompt.txt')

    with open(sample_prompts_path, 'w') as f:
        f.write(sample_prompts)

    run_cmd += f' --sample_sampler={sample_sampler}'
    run_cmd += f' --sample_prompts="{sample_prompts_path}"'

    if not sample_every_n_epochs == 0:
        run_cmd += f' --sample_every_n_epochs="{sample_every_n_epochs}"'

    if not sample_every_n_steps == 0:
        run_cmd += f' --sample_every_n_steps="{sample_every_n_steps}"'

    return run_cmd

def run_cmd_advanced_training(**kwargs):

    # awas note:
    # this is where defaults get set, so remove anything keys with value None
    # then the default getter with set them. hopefully thats right 
    to_del = []
    for k,v in kwargs.items():
        if v is None:
            to_del.append(k)

    for k in to_del:
        print("removing None val key %s" % (k))
        kwargs.pop(k)

    run_cmd = ''
    
    max_train_epochs = kwargs.get("max_train_epochs", "")
    if max_train_epochs:
        run_cmd += f' --max_train_epochs={max_train_epochs}'
        
    max_data_loader_n_workers = kwargs.get("max_data_loader_n_workers", "")
    if max_data_loader_n_workers:
        run_cmd += f' --max_data_loader_n_workers="{max_data_loader_n_workers}"'
    
    max_token_length = int(kwargs.get("max_token_length", 75))
    if max_token_length > 75:
        run_cmd += f' --max_token_length={max_token_length}'
        
    clip_skip = int(kwargs.get("clip_skip", 1))
    if clip_skip > 1:
        run_cmd += f' --clip_skip={clip_skip}'
        
    resume = kwargs.get("resume", "")
    if resume:
        run_cmd += f' --resume="{resume}"'
        
    keep_tokens = int(kwargs.get("keep_tokens", 0))
    if keep_tokens > 0:
        run_cmd += f' --keep_tokens="{keep_tokens}"'
        
    caption_dropout_every_n_epochs = int(kwargs.get("caption_dropout_every_n_epochs", 0))
    if caption_dropout_every_n_epochs > 0:
        run_cmd += f' --caption_dropout_every_n_epochs="{caption_dropout_every_n_epochs}"'
    
    caption_dropout_rate = float(kwargs.get("caption_dropout_rate", 0))
    if caption_dropout_rate > 0:
        run_cmd += f' --caption_dropout_rate="{caption_dropout_rate}"'
        
    vae_batch_size = int(kwargs.get("vae_batch_size", 0))
    if vae_batch_size > 0:
        run_cmd += f' --vae_batch_size="{vae_batch_size}"'
        
    bucket_reso_steps = int(kwargs.get("bucket_reso_steps", 64))
    run_cmd += f' --bucket_reso_steps={bucket_reso_steps}'
        
    save_every_n_steps = int(kwargs.get("save_every_n_steps", 0))
    if save_every_n_steps > 0:
        run_cmd += f' --save_every_n_steps="{save_every_n_steps}"'
        
    save_last_n_steps = int(kwargs.get("save_last_n_steps", 0))
    if save_last_n_steps > 0:
        run_cmd += f' --save_last_n_steps="{save_last_n_steps}"'
        
    save_last_n_steps_state = int(kwargs.get("save_last_n_steps_state", 0))
    if save_last_n_steps_state > 0:
        run_cmd += f' --save_last_n_steps_state="{save_last_n_steps_state}"'
        
    min_snr_gamma = int(kwargs.get("min_snr_gamma", 0))
    if min_snr_gamma >= 1:
        run_cmd += f' --min_snr_gamma={min_snr_gamma}'
    
    save_state = kwargs.get('save_state')
    if save_state:
        run_cmd += ' --save_state'
        
    mem_eff_attn = kwargs.get('mem_eff_attn')
    if mem_eff_attn:
        run_cmd += ' --mem_eff_attn'
    
    color_aug = kwargs.get('color_aug')
    if color_aug:
        run_cmd += ' --color_aug'
    
    flip_aug = kwargs.get('flip_aug')
    if flip_aug:
        run_cmd += ' --flip_aug'
    
    shuffle_caption = kwargs.get('shuffle_caption')
    if shuffle_caption:
        run_cmd += ' --shuffle_caption'
    
    gradient_checkpointing = kwargs.get('gradient_checkpointing')
    if gradient_checkpointing:
        run_cmd += ' --gradient_checkpointing'
    
    full_fp16 = kwargs.get('full_fp16')
    if full_fp16:
        run_cmd += ' --full_fp16'
    
    xformers = kwargs.get('xformers')
    if xformers:
        run_cmd += ' --xformers'
    
    persistent_data_loader_workers = kwargs.get('persistent_data_loader_workers')
    if persistent_data_loader_workers:
        run_cmd += ' --persistent_data_loader_workers'
    
    bucket_no_upscale = kwargs.get('bucket_no_upscale')
    if bucket_no_upscale:
        run_cmd += ' --bucket_no_upscale'
    
    random_crop = kwargs.get('random_crop')
    if random_crop:
        run_cmd += ' --random_crop'
        
    noise_offset_type = kwargs.get('noise_offset_type', 'Original')
    if noise_offset_type == 'Original':
        noise_offset = float(kwargs.get("noise_offset", 0))
        if noise_offset > 0:
            run_cmd += f' --noise_offset={noise_offset}'
        
        adaptive_noise_scale = float(kwargs.get("adaptive_noise_scale", 0))
        if adaptive_noise_scale != 0 and noise_offset > 0:
            run_cmd += f' --adaptive_noise_scale={adaptive_noise_scale}'
    else:
        multires_noise_iterations = int(kwargs.get("multires_noise_iterations", 0))
        if multires_noise_iterations > 0:
            run_cmd += f' --multires_noise_iterations="{multires_noise_iterations}"'
        
        multires_noise_discount = float(kwargs.get("multires_noise_discount", 0))
        if multires_noise_discount > 0:
            run_cmd += f' --multires_noise_discount="{multires_noise_discount}"'
    
    additional_parameters = kwargs.get("additional_parameters", "")
    if additional_parameters:
        run_cmd += f' {additional_parameters}'
    
    use_wandb = kwargs.get('use_wandb')
    if use_wandb:
        run_cmd += ' --log_with wandb'
    
    wandb_api_key = kwargs.get("wandb_api_key", "")
    if wandb_api_key:
        run_cmd += f' --wandb_api_key="{wandb_api_key}"'
        
    return run_cmd

def run_cmd_training(**kwargs):
    run_cmd = ''
    
    learning_rate = kwargs.get("learning_rate", "")
    if learning_rate:
        run_cmd += f' --learning_rate="{learning_rate}"'
    
    lr_scheduler = kwargs.get("lr_scheduler", "")
    if lr_scheduler:
        run_cmd += f' --lr_scheduler="{lr_scheduler}"'
    
    lr_warmup_steps = kwargs.get("lr_warmup_steps", "")
    if lr_warmup_steps:
        if lr_scheduler == 'constant':
            print('Can\'t use LR warmup with LR Scheduler constant... ignoring...')
        else:
            run_cmd += f' --lr_warmup_steps="{lr_warmup_steps}"'
    
    train_batch_size = kwargs.get("train_batch_size", "")
    if train_batch_size:
        run_cmd += f' --train_batch_size="{train_batch_size}"'
    
    max_train_steps = kwargs.get("max_train_steps", "")
    if max_train_steps:
        run_cmd += f' --max_train_steps="{max_train_steps}"'
    
    save_every_n_epochs = kwargs.get("save_every_n_epochs")
    if save_every_n_epochs:
        run_cmd += f' --save_every_n_epochs="{int(save_every_n_epochs)}"'
    
    mixed_precision = kwargs.get("mixed_precision", "")
    if mixed_precision:
        run_cmd += f' --mixed_precision="{mixed_precision}"'
    
    save_precision = kwargs.get("save_precision", "")
    if save_precision:
        run_cmd += f' --save_precision="{save_precision}"'
    
    seed = kwargs.get("seed", "")
    if seed != '':
        run_cmd += f' --seed="{seed}"'
    
    caption_extension = kwargs.get("caption_extension", "")
    if caption_extension:
        run_cmd += f' --caption_extension="{caption_extension}"'
    
    cache_latents = kwargs.get('cache_latents')
    if cache_latents:
        run_cmd += ' --cache_latents'
    
    cache_latents_to_disk = kwargs.get('cache_latents_to_disk')
    if cache_latents_to_disk:
        run_cmd += ' --cache_latents_to_disk'
    
    optimizer_type = kwargs.get("optimizer", "AdamW")
    run_cmd += f' --optimizer_type="{optimizer_type}"'
    
    optimizer_args = kwargs.get("optimizer_args", "")
    if optimizer_args != '':
        run_cmd += f' --optimizer_args {optimizer_args}'
    
    return run_cmd



def check_if_model_exist(output_name, output_dir, save_model_as, headless=False):
    p = os.path.join(output_dir, output_name + "." + save_model_as)
    if os.path.exists(p):
        if DEBUG:
            print("found already existing model %s" % (p))

        # deconflict name, figure it out later. lets just train shit
        garbage = str(random.random())[-8:]
        output_name = output_name + "_" + garbage

        p = os.path.join(output_dir, output_name + "." + save_model_as)
        if os.path.exists(p):
            raise ValueError("this no happen")

    return output_name, output_dir, save_model_as

def output_message(msg, headless=True):
    print(msg)

def open_configuration(
    file_path,
    ask_for_file=None,
    pretrained_model_name_or_path=None,
    v2=None,
    v_parameterization=None,
    logging_dir=None,
    train_data_dir=None,
    reg_data_dir=None,
    output_dir=None,
    max_resolution=None,
    learning_rate=None,
    lr_scheduler=None,
    lr_warmup=None,
    train_batch_size=None,
    epoch=None,
    save_every_n_epochs=None,
    mixed_precision=None,
    save_precision=None,
    seed=None,
    num_cpu_threads_per_process=None,
    cache_latents=None,
    cache_latents_to_disk=None,
    caption_extension=None,
    enable_bucket=None,
    gradient_checkpointing=None,
    full_fp16=None,
    no_token_padding=None,
    stop_text_encoder_training=None,
    # use_8bit_adam=None,
    xformers=None,
    save_model_as=None,
    shuffle_caption=None,
    save_state=None,
    resume=None,
    prior_loss_weight=None,
    text_encoder_lr=None,
    unet_lr=None,
    network_dim=None,
    lora_network_weights=None,
    dim_from_weights=None,
    color_aug=None,
    flip_aug=None,
    clip_skip=None,
    gradient_accumulation_steps=None,
    mem_eff_attn=None,
    output_name=None,
    model_list=None,
    max_token_length=None,
    max_train_epochs=None,
    max_data_loader_n_workers=None,
    network_alpha=None,
    training_comment=None,
    keep_tokens=None,
    lr_scheduler_num_cycles=None,
    lr_scheduler_power=None,
    persistent_data_loader_workers=None,
    bucket_no_upscale=None,
    random_crop=None,
    bucket_reso_steps=None,
    caption_dropout_every_n_epochs=None,
    caption_dropout_rate=None,
    optimizer=None,
    optimizer_args=None,
    noise_offset_type=None,
    noise_offset=None,
    adaptive_noise_scale=None,
    multires_noise_iterations=None,
    multires_noise_discount=None,
    LoRA_type=None,
    conv_dim=None,
    conv_alpha=None,
    sample_every_n_steps=None,
    sample_every_n_epochs=None,
    sample_sampler=None,
    sample_prompts=None,
    additional_parameters=None,
    vae_batch_size=None,
    min_snr_gamma=None,
    down_lr_weight=None,
    mid_lr_weight=None,
    up_lr_weight=None,
    block_lr_zero_threshold=None,
    block_dims=None,
    block_alphas=None,
    conv_dims=None,
    conv_alphas=None,
    weighted_captions=None,
    unit=None,
    save_every_n_steps=None,
    save_last_n_steps=None,
    save_last_n_steps_state=None,
    use_wandb=None,
    wandb_api_key=None
):
    # Get list of function parameters and values
    parameters = list(locals().items())

    if not file_path == '' and not file_path == None:
        # load variables from JSON file
        with open(file_path, 'r') as f:
            my_data = json.load(f)
            print('Loading config...')

            # Update values to fix deprecated use_8bit_adam checkbox, set appropriate optimizer if it is set to True, etc.
#            my_data = update_my_data(my_data)
    else:
        raise ValueError("im not reading this")

    values = [file_path]
    for key, value in parameters:
        # Set the value in the dictionary to the corresponding value in `my_data`, or the default value if not found
        if not key in ['ask_for_file', 'file_path']:
            values.append(my_data.get(key, value))

    return tuple(values)

def train_model(
    pretrained_model_name_or_path,
    v2,
    v_parameterization,
    logging_dir,
    train_data_dir,
    reg_data_dir,
    output_dir,
    max_resolution,
    learning_rate,
    lr_scheduler,
    lr_warmup,
    train_batch_size,
    epoch,
    save_every_n_epochs,
    mixed_precision,
    save_precision,
    seed,
    num_cpu_threads_per_process,
    cache_latents,
    cache_latents_to_disk,
    caption_extension,
    enable_bucket,
    gradient_checkpointing,
    full_fp16,
    no_token_padding,
    stop_text_encoder_training_pct,
    # use_8bit_adam,
    xformers,
    save_model_as,
    shuffle_caption,
    save_state,
    resume,
    prior_loss_weight,
    text_encoder_lr,
    unet_lr,
    network_dim,
    lora_network_weights,dim_from_weights,
    color_aug,
    flip_aug,
    clip_skip,
    gradient_accumulation_steps,
    mem_eff_attn,
    output_name,
    model_list,  # Keep this. Yes, it is unused here but required given the common list used
    max_token_length,
    max_train_epochs,
    max_data_loader_n_workers,
    network_alpha,
    training_comment,
    keep_tokens,
    lr_scheduler_num_cycles,
    lr_scheduler_power,
    persistent_data_loader_workers,
    bucket_no_upscale,
    random_crop,
    bucket_reso_steps,
    caption_dropout_every_n_epochs,
    caption_dropout_rate,
    optimizer,
    optimizer_args,
    noise_offset_type,noise_offset,adaptive_noise_scale,
    multires_noise_iterations,
    multires_noise_discount,
    LoRA_type,
    conv_dim,
    conv_alpha,
    sample_every_n_steps,
    sample_every_n_epochs,
    sample_sampler,
    sample_prompts,
    additional_parameters,
    vae_batch_size,
    min_snr_gamma,
    down_lr_weight,
    mid_lr_weight,
    up_lr_weight,
    block_lr_zero_threshold,
    block_dims,
    block_alphas,
    conv_dims,
    conv_alphas,
    weighted_captions,
    unit,
    save_every_n_steps,
    save_last_n_steps,
    save_last_n_steps_state,
    use_wandb,
    wandb_api_key,
):
    ## awas
    print_only_bool = args.print_only
    headless_bool = True 

    if pretrained_model_name_or_path == '':
        output_message(
            msg='Source model information is missing', headless=headless_bool
        )
        return

    if train_data_dir == '':
        output_message(
            msg='Image folder path is missing', headless=headless_bool
        )
        return

    if not os.path.exists(train_data_dir):
        output_message(
            msg='Image folder does not exist', headless=headless_bool
        )
        return

    if reg_data_dir != '':
        if not os.path.exists(reg_data_dir):
            output_message(
                msg='Regularisation folder does not exist',
                headless=headless_bool,
            )
            return

    if output_dir == '':
        output_message(
            msg='Output folder path is missing', headless=headless_bool
        )
        return

    if int(bucket_reso_steps) < 1:
        output_message(
            msg='Bucket resolution steps need to be greater than 0',
            headless=headless_bool,
        )
        return

    if noise_offset == '':
        noise_offset = 0

    if float(noise_offset) > 1 or float(noise_offset) < 0:
        output_message(
            msg='Noise offset need to be a value between 0 and 1',
            headless=headless_bool,
        )
        return

    # if float(noise_offset) > 0 and (
    #     multires_noise_iterations > 0 or multires_noise_discount > 0
    # ):
    #     output_message(
    #         msg="noise offset and multires_noise can't be set at the same time. Only use one or the other.",
    #         title='Error',
    #         headless=headless_bool,
    #     )
    #     return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if stop_text_encoder_training_pct > 0:
        output_message(
            msg='Output "stop text encoder training" is not yet supported. Ignoring',
            headless=headless_bool,
        )
        stop_text_encoder_training_pct = 0

    output_name, output_dir, save_model_as = check_if_model_exist(output_name, output_dir, save_model_as)

    if optimizer == 'Adafactor' and lr_warmup != '0':
        output_message(
            msg="Warning: lr_scheduler is set to 'Adafactor', so 'LR warmup (% of steps)' will be considered 0.",
            title='Warning',
            headless=headless_bool,
        )
        lr_warmup = '0'

    # If string is empty set string to 0.
    if text_encoder_lr == '':
        text_encoder_lr = 0
    if unet_lr == '':
        unet_lr = 0

    # Get a list of all subfolders in train_data_dir
    subfolders = [
        f
        for f in os.listdir(train_data_dir)
        if os.path.isdir(os.path.join(train_data_dir, f))
    ]

    total_steps = 0

    # Loop through each subfolder and extract the number of repeats
    for folder in subfolders:
        try:
            # Extract the number of repeats from the folder name
            repeats = int(folder.split('_')[0])

            # Count the number of images in the folder
            num_images = len(
                [
                    f
                    for f, lower_f in (
                        (file, file.lower())
                        for file in os.listdir(
                            os.path.join(train_data_dir, folder)
                        )
                    )
                    if lower_f.endswith(('.jpg', '.jpeg', '.png', '.webp'))
                ]
            )

            print(f'Folder {folder}: {num_images} images found')

            # Calculate the total number of steps for this folder
            steps = repeats * num_images

            # Print the result
            print(f'Folder {folder}: {steps} steps')

            total_steps += steps

        except ValueError:
            # Handle the case where the folder name does not contain an underscore
            print(
                f"Error: '{folder}' does not contain an underscore, skipping..."
            )

    if reg_data_dir == '':
        reg_factor = 1
    else:
        print(
            '\033[94mRegularisation images are used... Will double the number of steps required...\033[0m'
        )
        reg_factor = 2

    print(f'Total steps: {total_steps}')
    print(f'Train batch size: {train_batch_size}')
    print(f'Gradient accumulation steps: {gradient_accumulation_steps}')
    print(f'Epoch: {epoch}')
    print(f'Regulatization factor: {reg_factor}')

    # calculate max_train_steps
    max_train_steps = int(
        math.ceil(
            float(total_steps)
            / int(train_batch_size)
            / int(gradient_accumulation_steps)
            * int(epoch)
            * int(reg_factor)
        )
    )
    print(f'max_train_steps ({total_steps} / {train_batch_size} / {gradient_accumulation_steps} * {epoch} * {reg_factor}) = {max_train_steps}')

    # calculate stop encoder training
    if stop_text_encoder_training_pct == None:
        stop_text_encoder_training = 0
    else:
        stop_text_encoder_training = math.ceil(
            float(max_train_steps) / 100 * int(stop_text_encoder_training_pct)
        )
    print(f'stop_text_encoder_training = {stop_text_encoder_training}')

    lr_warmup_steps = round(float(int(lr_warmup) * int(max_train_steps) / 100))
    print(f'lr_warmup_steps = {lr_warmup_steps}')

    run_cmd = f'accelerate launch --num_cpu_threads_per_process={num_cpu_threads_per_process} "train_network.py"'

    if v2:
        run_cmd += ' --v2'
    if v_parameterization:
        run_cmd += ' --v_parameterization'
    if enable_bucket:
        run_cmd += ' --enable_bucket'
    if no_token_padding:
        run_cmd += ' --no_token_padding'
    if weighted_captions:
        run_cmd += ' --weighted_captions'
    run_cmd += (
        f' --pretrained_model_name_or_path="{pretrained_model_name_or_path}"'
    )
    run_cmd += f' --train_data_dir="{train_data_dir}"'
    if len(reg_data_dir):
        run_cmd += f' --reg_data_dir="{reg_data_dir}"'
    run_cmd += f' --resolution={max_resolution}'
    run_cmd += f' --output_dir="{output_dir}"'
    if not logging_dir == '':
        run_cmd += f' --logging_dir="{logging_dir}"'
    run_cmd += f' --network_alpha="{network_alpha}"'
    if not training_comment == '':
        run_cmd += f' --training_comment="{training_comment}"'
    if not stop_text_encoder_training == 0:
        run_cmd += (
            f' --stop_text_encoder_training={stop_text_encoder_training}'
        )
    if not save_model_as == 'same as source model':
        run_cmd += f' --save_model_as={save_model_as}'
    if not float(prior_loss_weight) == 1.0:
        run_cmd += f' --prior_loss_weight={prior_loss_weight}'
    if LoRA_type == 'LoCon' or LoRA_type == 'LyCORIS/LoCon':
        try:
            import lycoris
        except ModuleNotFoundError:
            print(
                "\033[1;31mError:\033[0m The required module 'lycoris_lora' is not installed. Please install by running \033[33mupgrade.ps1\033[0m before running this program."
            )
            return
        run_cmd += f' --network_module=lycoris.kohya'
        run_cmd += f' --network_args "conv_dim={conv_dim}" "conv_alpha={conv_alpha}" "algo=lora"'
    if LoRA_type == 'LyCORIS/LoHa':
        try:
            import lycoris
        except ModuleNotFoundError:
            print(
                "\033[1;31mError:\033[0m The required module 'lycoris_lora' is not installed. Please install by running \033[33mupgrade.ps1\033[0m before running this program."
            )
            return
        run_cmd += f' --network_module=lycoris.kohya'
        run_cmd += f' --network_args "conv_dim={conv_dim}" "conv_alpha={conv_alpha}" "algo=loha"'

    if LoRA_type in ['Kohya LoCon', 'Standard']:
        kohya_lora_var_list = [
            'down_lr_weight',
            'mid_lr_weight',
            'up_lr_weight',
            'block_lr_zero_threshold',
            'block_dims',
            'block_alphas',
            'conv_dims',
            'conv_alphas',
        ]

        run_cmd += f' --network_module=networks.lora'
        kohya_lora_vars = {
            key: value
            for key, value in vars().items()
            if key in kohya_lora_var_list and value
        }

        network_args = ''
        if LoRA_type == 'Kohya LoCon':
            network_args += f' "conv_dim={conv_dim}" "conv_alpha={conv_alpha}"'

        for key, value in kohya_lora_vars.items():
            if value:
                network_args += f' {key}="{value}"'

        if network_args:
            run_cmd += f' --network_args{network_args}'

    if LoRA_type in ['Kohya DyLoRA']:
        kohya_lora_var_list = [
            'conv_dim',
            'conv_alpha',
            'down_lr_weight',
            'mid_lr_weight',
            'up_lr_weight',
            'block_lr_zero_threshold',
            'block_dims',
            'block_alphas',
            'conv_dims',
            'conv_alphas',
            'unit',
        ]

        run_cmd += f' --network_module=networks.dylora'
        kohya_lora_vars = {
            key: value
            for key, value in vars().items()
            if key in kohya_lora_var_list and value
        }

        network_args = ''

        for key, value in kohya_lora_vars.items():
            if value:
                network_args += f' {key}="{value}"'

        if network_args:
            run_cmd += f' --network_args{network_args}'

    if not (float(text_encoder_lr) == 0) or not (float(unet_lr) == 0):
        if not (float(text_encoder_lr) == 0) and not (float(unet_lr) == 0):
            run_cmd += f' --text_encoder_lr={text_encoder_lr}'
            run_cmd += f' --unet_lr={unet_lr}'
        elif not (float(text_encoder_lr) == 0):
            run_cmd += f' --text_encoder_lr={text_encoder_lr}'
            run_cmd += f' --network_train_text_encoder_only'
        else:
            run_cmd += f' --unet_lr={unet_lr}'
            run_cmd += f' --network_train_unet_only'
    else:
        if float(learning_rate) == 0:
            output_message(
                msg='Please input learning rate values.',
                headless=headless_bool,
            )
            return

    run_cmd += f' --network_dim={network_dim}'

    if LoRA_type not in ['LyCORIS/LoCon', 'LyCORIS/LoHa']:
        if not lora_network_weights == '':
            run_cmd += f' --network_weights="{lora_network_weights}"'
        if dim_from_weights:
            run_cmd += f' --dim_from_weights'
            
    if int(gradient_accumulation_steps) > 1:
        run_cmd += f' --gradient_accumulation_steps={int(gradient_accumulation_steps)}'
    if not output_name == '':
        run_cmd += f' --output_name="{output_name}"'
    if not lr_scheduler_num_cycles == '':
        run_cmd += f' --lr_scheduler_num_cycles="{lr_scheduler_num_cycles}"'
    else:
        run_cmd += f' --lr_scheduler_num_cycles="{epoch}"'
    if not lr_scheduler_power == '':
        run_cmd += f' --lr_scheduler_power="{lr_scheduler_power}"'

    run_cmd += run_cmd_training(
        learning_rate=learning_rate,
        lr_scheduler=lr_scheduler,
        lr_warmup_steps=lr_warmup_steps,
        train_batch_size=train_batch_size,
        max_train_steps=max_train_steps,
        save_every_n_epochs=save_every_n_epochs,
        mixed_precision=mixed_precision,
        save_precision=save_precision,
        seed=seed,
        caption_extension=caption_extension,
        cache_latents=cache_latents,
        cache_latents_to_disk=cache_latents_to_disk,
        optimizer=optimizer,
        optimizer_args=optimizer_args,
    )

    run_cmd += run_cmd_advanced_training(
        max_train_epochs=max_train_epochs,
        max_data_loader_n_workers=max_data_loader_n_workers,
        max_token_length=max_token_length,
        resume=resume,
        save_state=save_state,
        mem_eff_attn=mem_eff_attn,
        clip_skip=clip_skip,
        flip_aug=flip_aug,
        color_aug=color_aug,
        shuffle_caption=shuffle_caption,
        gradient_checkpointing=gradient_checkpointing,
        full_fp16=full_fp16,
        xformers=xformers,
        # use_8bit_adam=use_8bit_adam,
        keep_tokens=keep_tokens,
        persistent_data_loader_workers=persistent_data_loader_workers,
        bucket_no_upscale=bucket_no_upscale,
        random_crop=random_crop,
        bucket_reso_steps=bucket_reso_steps,
        caption_dropout_every_n_epochs=caption_dropout_every_n_epochs,
        caption_dropout_rate=caption_dropout_rate,
        noise_offset_type=noise_offset_type,
        noise_offset=noise_offset,
        adaptive_noise_scale=adaptive_noise_scale,
        multires_noise_iterations=multires_noise_iterations,
        multires_noise_discount=multires_noise_discount,
        additional_parameters=additional_parameters,
        vae_batch_size=vae_batch_size,
        min_snr_gamma=min_snr_gamma,
        save_every_n_steps=save_every_n_steps,
        save_last_n_steps=save_last_n_steps,
        save_last_n_steps_state=save_last_n_steps_state,
        use_wandb=use_wandb,
        wandb_api_key=wandb_api_key,
    )

    run_cmd += run_cmd_sample(
        sample_every_n_steps,
        sample_every_n_epochs,
        sample_sampler,
        sample_prompts,
        output_dir,
    )

    # if not down_lr_weight == '':
    #     run_cmd += f' --down_lr_weight="{down_lr_weight}"'
    # if not mid_lr_weight == '':
    #     run_cmd += f' --mid_lr_weight="{mid_lr_weight}"'
    # if not up_lr_weight == '':
    #     run_cmd += f' --up_lr_weight="{up_lr_weight}"'
    # if not block_lr_zero_threshold == '':
    #     run_cmd += f' --block_lr_zero_threshold="{block_lr_zero_threshold}"'
    # if not block_dims == '':
    #     run_cmd += f' --block_dims="{block_dims}"'
    # if not block_alphas == '':
    #     run_cmd += f' --block_alphas="{block_alphas}"'
    # if not conv_dims == '':
    #     run_cmd += f' --conv_dims="{conv_dims}"'
    # if not conv_alphas == '':
    #     run_cmd += f' --conv_alphas="{conv_alphas}"'

    if print_only_bool:
        print(
            '\033[93m\nHere is the trainer command as a reference. It will not be executed:\033[0m\n'
        )
        print('\033[96m' + run_cmd + '\033[0m\n')
    else:
        print(run_cmd)
        # Run the command
        if os.name == 'posix':
            os.system(run_cmd)
        else:
            subprocess.run(run_cmd)

def find_directories(args):

    if args.base_dir.endswith("*"):
        base_dir = args.base_dir
    elif args.base_dir.endswith("/"):
        base_dir = args.base_dir + "*"
    else:
        base_dir = args.base_dir  + "/*"

    print("searching in %s" % (base_dir))
    raw_lora_dirs = glob.glob(base_dir)
    print("found %d possible lora subdirs" % (len(raw_lora_dirs)))

    
    lora_dirs = []
    for ld in raw_lora_dirs:
        ld_basename = os.path.basename(ld)
        if regex.search(args.match_regex, ld_basename):
            if DEBUG:
                print("directory %s matched match_regex %s" % (ld_basename, args.match_regex))

            if args.exclude_regex and regex.search(args.exclude_regex, ld_basename):
                if DEBUG:
                    print("directory %s excluded as matched exclude_regex %s" % (ld_basename, args.exclude_regex))
            else:
                lora_dirs.append(ld)
#        else:
#            if DEBUG:
#                print("directory %s did not  match_regex %s" % (ld_basename, args.match_regex))

    print("filtered to %d lora subdirs" % (len(lora_dirs)))
    return lora_dirs


# given a lora_dir, find its config file. 
# we dont necessarily know the name, just find a json 
# file, if we find multiple, then implode.
def locate_config_file(lora_dir):
    files = glob.glob(lora_dir + "/*.json")
    if len(files) == 0:
        raise SomethinError("Could not find .json config file in subdirectory '%s'" % (lora_dir))
    elif len(files) > 1:
        raise SomethinError("Found multiple .json config files in subdirectory '%s', not sure what to do" % (lora_dir))
    else:
        return files[0]

# check if a passed lora_dir is in the shape we expect.
# for me, this means it has a config that ends in json
# then we load that
#def validate_directory(lora_dir):


def main(args):

    lora_dirs = find_directories(args)
    if len(lora_dirs) == 0:
        raise ValueError("did not match any directories")

    for lora_dir in lora_dirs:
        config_file = locate_config_file(lora_dir)
        
        if DEBUG:
            print("using config_file %s" % (config_file))

        fucken = open_configuration(config_file)
        print(fucken[1:])
        train_model(*fucken[1:])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="base directory, containing one subdirectory per lora")
    parser.add_argument("--match_regex", type=str, required=False, default=".*", help="apply this regex pattern, subdirectories that match will be included")
    parser.add_argument("--exclude_regex", type=str, required=False, help="apply this regex pattern, subdirectories that match  will be excluded")
    parser.add_argument("--print_only", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    DEBUG = args.debug

    main(args)
