import os
import cv2
import einops
import numpy as np
import torch
import torch.optim as optim
import random
import imageio
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import scipy.interpolate
from pytorch_lightning import seed_everything

from .annotator.util import resize_image, HWC3
from .annotator.canny import CannyDetector
from .annotator.midas import MidasDetector
from .cldm.model import create_model, load_state_dict
from .ldm.models.diffusion.ddim import DDIMSampler
from .atlas_data import AtlasData
from .atlas_utils import get_grid_indices, get_atlas_bounding_box
from .aggnet import AGGNet


class StableVideo:
    def __init__(self, base_cfg, canny_model_cfg, depth_model_cfg, save_gpu_memory=False, device="cpu"):
        self.base_cfg = base_cfg
        self.canny_model_cfg = canny_model_cfg
        self.depth_model_cfg = depth_model_cfg
        self.img2img_model = None
        self.canny_model = None
        self.depth_model = None
        self.b_atlas = None
        self.f_atlas = None
        self.data = None
        self.crops = None
        self.save_gpu_memory = save_gpu_memory
        self.device = device
    
    def load_canny_model(
        self,
        base_cfg='ckpt/cldm_v15.yaml',
        canny_model_cfg='ckpt/control_sd15_canny.pth', 
    ):
        self.apply_canny = CannyDetector()
        canny_model = create_model(base_cfg).cpu()
        canny_model.load_state_dict(load_state_dict(canny_model_cfg, location=self.device), strict=False)
        self.canny_ddim_sampler = DDIMSampler(canny_model)
        self.canny_model = canny_model
        
    def load_depth_model(
        self,
        base_cfg='ckpt/cldm_v15.yaml',
        depth_model_cfg='ckpt/control_sd15_depth.pth',
    ):
        self.apply_midas = MidasDetector(device=self.device)
        depth_model = create_model(base_cfg).cpu()
        depth_model.load_state_dict(load_state_dict(depth_model_cfg, location=self.device), strict=False)
        self.depth_ddim_sampler = DDIMSampler(depth_model)
        self.depth_model = depth_model

    def load_video(self, video_name):
        self.data = AtlasData(video_name, self.device)
        save_name = f"data/{video_name}/{video_name}.mp4"
        if not os.path.exists(save_name):
            imageio.mimwrite(save_name, self.data.original_video.cpu().permute(0, 2, 3, 1))
            print("original video saved.")
        toIMG = transforms.ToPILImage()
        self.f_atlas_origin = toIMG(self.data.cropped_foreground_atlas[0])
        self.b_atlas_origin = toIMG(self.data.background_grid_atlas[0])
        return save_name, self.f_atlas_origin, self.b_atlas_origin
    
    @torch.no_grad()
    def edit_background(
            self, 
            prompt="", 
            a_prompt="best quality, extremely detailed", 
            n_prompt="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",  
            image_resolution=512, 
            detect_resolution=384,
            ddim_steps=20, 
            scale=9, 
            seed=-1, 
            eta=0,
            num_samples=1):
        
        # Restore to GPU if model was unloaded to save memory
        if self.save_gpu_memory and self.device != "cpu":
            self.depth_model = self.depth_model.to(torch.device(self.device))
        
        input_image = self.b_atlas_origin
        size = input_image.size
        ddim_sampler = self.depth_ddim_sampler
        apply_midas = self.apply_midas
        
        input_image = np.array(input_image)
        input_image = HWC3(input_image)
        detected_map, _ = apply_midas(resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().to(torch.device(self.device)) / 255.0
        control = torch.stack([control for _ in range(1)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        cond = {"c_concat": [control], "c_crossattn": [self.depth_model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": [control], "c_crossattn": [self.depth_model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)
    

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)
        
        x_samples = self.depth_model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
        self.b_atlas = Image.fromarray(results[0]).resize(size)
        
        # Unload from GPU to save memory
        if self.save_gpu_memory and self.device != "cpu":
            self.depth_model = self.depth_model.cpu()
        
        return self.b_atlas
    
    @torch.no_grad()
    def advanced_edit_foreground(
            self, 
            keyframes="0", 
            res=2000,
            prompt="", 
            a_prompt="best quality, extremely detailed", 
            n_prompt="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",  
            image_resolution=512, 
            low_threshold=100, 
            high_threshold=200,
            ddim_steps=20,
            s=0.9,
            scale=9, 
            seed=-1, 
            eta=0,
            if_net=False,
            num_samples=1):
        
        # Restore to GPU if model was unloaded to save memory
        if self.save_gpu_memory and self.device != "cpu":
            self.canny_model = self.canny_model.to(torch.device(self.device))
        
        keyframes = [int(x) for x in keyframes.split(",")]
        if self.data is None:
            raise ValueError("Please load video first")
        self.crops = self.data.get_global_crops_multi(keyframes, res)
        n_keyframes = len(keyframes)
        indices = get_grid_indices(0, 0, res, res)
        f_atlas = torch.zeros(size=(n_keyframes, res, res, 3,)).to(torch.device(self.device))

        img_list = [transforms.ToPILImage()(i[0]) for i in self.crops['original_foreground_crops']]
        result_list = []
        
        # initial setting
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)
        
        self.canny_ddim_sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=eta, verbose=False)
        c_crossattn = [self.canny_model.get_learned_conditioning([prompt + ', ' + a_prompt])]
        uc_crossattn = [self.canny_model.get_learned_conditioning([n_prompt])]
        
        for i in range(n_keyframes):
            # get current keyframe
            current_img = img_list[i]
            img = resize_image(HWC3(np.array(current_img)), image_resolution)
            H, W, C = img.shape
            shape = (4, H // 8, W // 8)
            # get canny control
            detected_map = self.apply_canny(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)
            control = torch.from_numpy(detected_map.copy()).float().to(torch.device(self.device)) / 255.0
            control = einops.rearrange(control.unsqueeze(0), 'b h w c -> b c h w').clone()
            
            cond = {"c_concat": [control], "c_crossattn": c_crossattn}
            un_cond = {"c_concat": [control], "c_crossattn": uc_crossattn}
            
            
            # if not the key frame, calculate the mapping from last atlas
            if i == 0:
                latent = torch.randn((1, 4, H // 8, W // 8)).to(torch.device(self.device))
                samples, _ = self.canny_ddim_sampler.sample(ddim_steps, num_samples,
                                                            shape, cond, verbose=False, eta=eta,
                                                            unconditional_guidance_scale=scale,
                                                            unconditional_conditioning=un_cond,
                                                            x_T=latent)
            else:
                last_atlas = f_atlas[i-1:i].permute(0, 3, 2, 1)
                mapped_img = F.grid_sample(last_atlas, self.crops['foreground_uvs'][i].reshape(1, -1, 1, 2), mode="bilinear", align_corners=self.data.config["align_corners"]).clamp(min=0.0, max=1.0).reshape((3, current_img.size[1], current_img.size[0]))
                mapped_img = transforms.ToPILImage()(mapped_img)
                
                mapped_img = mapped_img.resize((W, H))
                mapped_img = np.array(mapped_img).astype(np.float32) / 255.0
                mapped_img = mapped_img[None].transpose(0, 3, 1, 2)
                mapped_img = torch.from_numpy(mapped_img).to(torch.device(self.device))
                mapped_img = 2. * mapped_img - 1.
                latent = self.canny_model.get_first_stage_encoding(self.canny_model.encode_first_stage(mapped_img))
                
                t_enc = int(ddim_steps * s)
                latent = self.canny_ddim_sampler.stochastic_encode(latent, torch.tensor([t_enc]).to(torch.device(self.device)))
                samples = self.canny_ddim_sampler.decode(x_latent=latent, 
                                                         cond=cond, 
                                                         t_start=t_enc,
                                                         unconditional_guidance_scale=scale,
                                                         unconditional_conditioning=un_cond)

            x_samples = self.canny_model.decode_first_stage(samples)
            result = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            result = Image.fromarray(result[0])
            
            result = result.resize(current_img.size)
            result = transforms.ToTensor()(result)
            # times alpha
            alpha = self.crops['foreground_alpha'][i][0].cpu()
            result = alpha * result
            
            # buffer for training
            result_copy = result.clone().to(torch.device(self.device))
            result_copy.requires_grad = True
            result_list.append(result_copy)
            
            # map to atlas
            uv = (self.crops['foreground_uvs'][i].reshape(-1, 2) * 0.5 + 0.5) * res
            for c in range(3):
                interpolated = scipy.interpolate.griddata(
                    points=uv.cpu().numpy(),
                    values=result[c].reshape(-1, 1).cpu().numpy(),
                    xi=indices.reshape(-1, 2).cpu().numpy(),
                    method="linear",
                ).reshape(res, res)
                interpolated = torch.from_numpy(interpolated).float()
                interpolated[interpolated.isnan()] = 0.0
                f_atlas[i, :, :, c] = interpolated

        f_atlas = f_atlas.permute(0, 3, 2, 1)
        
        # aggregate via simple median as begining
        agg_atlas, _ = torch.median(f_atlas, dim=0)
        
        if if_net == True:
            #####################################
            #           aggregate net           #
            #####################################
            lr, n_epoch = 1e-3, 500
            agg_net = AGGNet().to(torch.device(self.device))
            loss_fn = nn.L1Loss()
            optimizer = optim.SGD(agg_net.parameters(), lr=lr, momentum=0.9)
            for _ in range(n_epoch):
                loss = 0.
                for i in range(n_keyframes):
                    e_img = result_list[i]
                    temp_agg_atlas = agg_net(agg_atlas)
                    rec_img = F.grid_sample(temp_agg_atlas[None], 
                                            self.crops['foreground_uvs'][i].reshape(1, -1, 1, 2), 
                                            mode="bilinear", 
                                            align_corners=self.data.config["align_corners"])
                    rec_img = rec_img.clamp(min=0.0, max=1.0).reshape(e_img.shape)
                    loss += loss_fn(rec_img, e_img)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            agg_atlas = agg_net(agg_atlas)
        #####################################
        
        agg_atlas, _ = get_atlas_bounding_box(self.data.mask_boundaries, agg_atlas, self.data.foreground_all_uvs)
        self.f_atlas = transforms.ToPILImage()(agg_atlas)
        
        # Unload from GPU to save memory
        if self.save_gpu_memory and self.device != "cpu":
            self.canny_model = self.canny_model.cpu()
        
        return self.f_atlas

    @torch.no_grad()
    def render(self, f_atlas, f_mask, b_atlas):
        # foreground
        if f_atlas == None:
            f_atlas = transforms.ToTensor()(self.f_atlas_origin).unsqueeze(0).to(torch.device(self.device))
        else:
            f_atlas_origin = transforms.ToTensor()(self.f_atlas_origin).unsqueeze(0).to(torch.device(self.device))
            f_atlas = transforms.ToTensor()(f_atlas).unsqueeze(0).to(torch.device(self.device))
            f_mask = transforms.ToTensor()(f_mask).unsqueeze(0).to(torch.device(self.device))
            if f_atlas.shape != f_mask.shape:
                print("Warning: truncating mask to atlas shape {}".format(f_atlas.shape))
                f_mask = f_mask[:f_atlas.shape[0], :f_atlas.shape[1], :f_atlas.shape[2], :f_atlas.shape[3]]
            f_atlas = f_atlas * (1 - f_mask) + f_atlas_origin * f_mask
        
        f_atlas = torch.nn.functional.pad(
            f_atlas,
            pad=(
                self.data.foreground_atlas_bbox[1],
                self.data.foreground_grid_atlas.shape[-1] - (self.data.foreground_atlas_bbox[1] + self.data.foreground_atlas_bbox[3]),
                self.data.foreground_atlas_bbox[0],
                self.data.foreground_grid_atlas.shape[-2] - (self.data.foreground_atlas_bbox[0] + self.data.foreground_atlas_bbox[2]),
            ),
            mode="replicate",
        )
        foreground_edit = F.grid_sample(
            f_atlas, self.data.scaled_foreground_uvs, mode="bilinear", align_corners=self.data.config["align_corners"]
        ).clamp(min=0.0, max=1.0)
        
        foreground_edit = foreground_edit.squeeze().t()  # shape (batch, 3)
        foreground_edit = (
            foreground_edit.reshape(self.data.config["maximum_number_of_frames"], self.data.config["resy"], self.data.config["resx"], 3)
            .permute(0, 3, 1, 2)
            .clamp(min=0.0, max=1.0)
        )
        # background
        if b_atlas == None:
            b_atlas = self.b_atlas_origin

        b_atlas = transforms.ToTensor()(b_atlas).unsqueeze(0).to(torch.device(self.device))
        background_edit = F.grid_sample(
            b_atlas, self.data.scaled_background_uvs, mode="bilinear", align_corners=self.data.config["align_corners"]
        ).clamp(min=0.0, max=1.0)
        background_edit = background_edit.squeeze().t()  # shape (batch, 3)
        background_edit = (
            background_edit.reshape(self.data.config["maximum_number_of_frames"], self.data.config["resy"], self.data.config["resx"], 3)
            .permute(0, 3, 1, 2)
            .clamp(min=0.0, max=1.0)
        )
        
        output_video = 255 * ((self.data.all_alpha * foreground_edit) + ((1 - self.data.all_alpha) * background_edit))
        output_video = output_video.detach().cpu()
        output_video = output_video.to(torch.uint8).permute(0, 2, 3, 1)
        
        return output_video
