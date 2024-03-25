import os
import torch

from stablevideo import StableVideo

if __name__ == '__main__':
    import gradio as gr

    with torch.cuda.amp.autocast():
        stablevideo = StableVideo(base_cfg="ckpt/cldm_v15.yaml",
                                canny_model_cfg="ckpt/control_sd15_canny.pth",
                                depth_model_cfg="ckpt/control_sd15_depth.pth",
                                save_memory=True)

        stablevideo.load_canny_model()
        stablevideo.load_depth_model()
        
        block = gr.Blocks().queue()
        with block:
            with gr.Row():
                gr.Markdown("## StableVideo")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Select one example video and click **Load Video** buttom and wait for 10 sec.")
                    original_video = gr.Video(label="Original Video", interactive=False, height=432, width=768)
                    with gr.Row():
                        foreground_atlas = gr.Image(label="Foreground Atlas", type="pil", height=216, width=384)
                        background_atlas = gr.Image(label="Background Atlas", type="pil", height=216, width=384)
                    load_video_button = gr.Button("Load Video")
                    avail_video = [f.name for f in os.scandir("data") if f.is_dir()]
                    video_name = gr.Radio(choices=avail_video,
                                        label="Select Example Videos",
                                        value="car-turn")
                with gr.Column():
                    gr.Markdown("### Write text prompt and advanced options for background and foreground. Click render.")
                    output_video = gr.Video(label="Output Video", interactive=False, height=432, width=768)
                    with gr.Row():
                        output_foreground_atlas = gr.ImageMask(label="Editable Output Foreground Atlas", type="pil", tool="sketch", interactive=True, height=216, width=384)
                        output_background_atlas = gr.Image(label="Output Background Atlas", type="pil", interactive=False, height=216, width=384)
                    run_button = gr.Button("Render")
                    with gr.Row():
                        with gr.Column():
                            f_advance_run_button = gr.Button("Advanced Edit Foreground")
                            f_prompt = gr.Textbox(label="Foreground Prompt", value="a picture of an orange suv")
                            with gr.Accordion("Advanced Foreground Options", open=False):
                                    adv_keyframes = gr.Textbox(label="keyframe", value="20, 40, 60")
                                    adv_atlas_resolution = gr.Slider(label="Atlas Resolution", minimum=1000, maximum=3000, value=2000, step=100)
                                    adv_image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=256)
                                    adv_low_threshold = gr.Slider(label="Canny low threshold", minimum=1, maximum=255, value=100, step=1)
                                    adv_high_threshold = gr.Slider(label="Canny high threshold", minimum=1, maximum=255, value=200, step=1)
                                    adv_ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                                    adv_s = gr.Slider(label="Noise Scale", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                                    adv_scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=15.0, value=9.0, step=0.1)
                                    adv_seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                                    adv_eta = gr.Number(label="eta (DDIM)", value=0.0)
                                    adv_a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed, no background')
                                    adv_n_prompt = gr.Textbox(label="Negative Prompt", value='lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
                                    adv_if_net = gr.gradio.Checkbox(label="if use agg net", value=False)
                        
                        with gr.Column():
                            b_run_button = gr.Button("Edit Background")
                            b_prompt = gr.Textbox(label="Background Prompt", value="winter scene, snowy scene, beautiful snow")
                            with gr.Accordion("Background Options", open=False):
                                b_image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=256)
                                b_detect_resolution = gr.Slider(label="Depth Resolution", minimum=128, maximum=1024, value=512, step=1)
                                b_ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                                b_scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                                b_seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                                b_eta = gr.Number(label="eta (DDIM)", value=0.0)
                                b_a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                                b_n_prompt = gr.Textbox(label="Negative Prompt", value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
                    
                        
                        
            # edit param
            f_adv_edit_param = [adv_keyframes, 
                                adv_atlas_resolution, 
                                f_prompt, 
                                adv_a_prompt, 
                                adv_n_prompt, 
                                adv_image_resolution, 
                                adv_low_threshold, 
                                adv_high_threshold, 
                                adv_ddim_steps, 
                                adv_s,
                                adv_scale, 
                                adv_seed, 
                                adv_eta,
                                adv_if_net]
            b_edit_param = [b_prompt, 
                            b_a_prompt, 
                            b_n_prompt, 
                            b_image_resolution, 
                            b_detect_resolution, 
                            b_ddim_steps, 
                            b_scale, 
                            b_seed,
                            b_eta]

            load_video_button.click(fn=stablevideo.load_video, inputs=video_name, outputs=[original_video, foreground_atlas, background_atlas])
            f_advance_run_button.click(fn=stablevideo.advanced_edit_foreground, inputs=f_adv_edit_param, outputs=[output_foreground_atlas])
            b_run_button.click(fn=stablevideo.edit_background, inputs=b_edit_param, outputs=[output_background_atlas])
            run_button.click(fn=stablevideo.render, inputs=[output_foreground_atlas, output_background_atlas], outputs=[output_video])
        
        block.launch(share=True)