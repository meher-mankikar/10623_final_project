# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
from models.vae_adain import Model as VAE
from models.latent_points_ada_localprior import PVCNN2Prior as LocalPrior
from utils.diffusion_pvd import DiffusionDiscretized
from utils.vis_helper import plot_points
from utils.model_helper import import_model
from diffusers import DDPMScheduler
import torch
from matplotlib import pyplot as plt

from models.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver

class LION(object):
    def __init__(self, cfg):
        self.vae = VAE(cfg).cuda()
        GlobalPrior = import_model(cfg.latent_pts.style_prior)
        global_prior = GlobalPrior(cfg.sde, cfg.latent_pts.style_dim, cfg).cuda()
        local_prior = LocalPrior(cfg.sde, cfg.shapelatent.latent_dim, cfg).cuda()   # PVCNN2Prior
        self.priors = torch.nn.ModuleList([global_prior, local_prior])
        self.scheduler = DDPMScheduler(clip_sample=False,
                                       beta_start=cfg.ddpm.beta_1, beta_end=cfg.ddpm.beta_T, beta_schedule=cfg.ddpm.sched_mode,
                                       num_train_timesteps=cfg.ddpm.num_steps, variance_type=cfg.ddpm.model_var_type)
        self.diffusion = DiffusionDiscretized(None, None, cfg)

        # augment with DPM
        self.cfg = cfg
        self.dpm_steps = cfg.ddpm.ddim_step

        # self.load_model(cfg)

    def load_model(self, model_path):
        # model_path = cfg.ckpt.path
        ckpt = torch.load(model_path)
        self.priors.load_state_dict(ckpt['dae_state_dict'])
        self.vae.load_state_dict(ckpt['vae_state_dict'])
        print(f'INFO finish loading from {model_path}')

    @torch.no_grad()
    def sample(self, num_samples=10, clip_feat=None, save_img=False, use_dpm_solver=False):
        self.scheduler.set_timesteps(1000, device='cuda')
        timesteps = self.scheduler.timesteps
        latent_shape = self.vae.latent_shape()
        global_prior, local_prior = self.priors[0], self.priors[1]
        assert(not local_prior.mixed_prediction and not global_prior.mixed_prediction)
        sampled_list = []
        output_dict = {}

        # start sample global prior (p_theta(z) = distribution of global shape latent DDM)
        x_T_shape = [num_samples] + latent_shape[0]
        x_noisy = torch.randn(size=x_T_shape, device='cuda')
        condition_input = None


        ###################################
        for i, t in enumerate(timesteps):
            t_tensor = torch.ones(num_samples, dtype=torch.int64, device='cuda') * (t+1)
            # GlobalPrior Layer: 
            noise_pred = global_prior(x=x_noisy, t=t_tensor.float(), 
                    condition_input=condition_input, clip_feat=clip_feat)
            x_noisy = self.scheduler.step(noise_pred, t, x_noisy).prev_sample

        ###################################
        # self.dpm_noise_schedule = NoiseScheduleVP(
        #     schedule=self.cfg.ddpm.sched_mode,
        #     continuous_beta_0=self.cfg.ddpm.beta_1,
        #     continuous_beta_1=self.cfg.ddpm.beta_T
        # )

        # self.dpm_global_prior = model_wrapper(
        #     global_prior,
        #     self.dpm_noise_schedule,
        #     model_type="noise",
        #     model_kwargs={"clip_feat": clip_feat}
        # )

        # self.dpm_global_solver = DPM_Solver(
        #     self.dpm_global_prior,
        #     self.dpm_noise_schedule,
        #     algorithm_type="dpmsolver"
        # )

        # x_noisy = self.dpm_global_solver.sample(
        #     x_noisy,
        #     steps=self.dpm_steps,
        #     order=3,
        #     skip_type="time_uniform",
        #     method="singlestep",
        # )
        ###################################

        sampled_list.append(x_noisy)
        output_dict['z_global'] = x_noisy

        condition_input = x_noisy
        condition_input = self.vae.global2style(condition_input)

        # # start sample local prior (p_phi(h | z) = DDM modeling the piont cloud-structured latents)
        x_T_shape = [num_samples] + latent_shape[1]
        x_noisy = torch.randn(size=x_T_shape, device='cuda')

        ###################################
        for i, t in enumerate(timesteps):
            t_tensor = torch.ones(num_samples, dtype=torch.int64, device='cuda') * (t+1)

            # Input: x: B,ND or B,ND,1,1
            noise_pred = local_prior(x=x_noisy, t=t_tensor.float(), 
                   condition_input=condition_input, clip_feat=clip_feat)

            x_noisy = self.scheduler.step(noise_pred, t, x_noisy).prev_sample

        ###################################
        #self.dpm_noise_schedule = NoiseScheduleVP(
        #    schedule=self.cfg.ddpm.sched_mode,
        #    continuous_beta_0=self.cfg.ddpm.beta_1,
        #    continuous_beta_1=self.cfg.ddpm.beta_T
        #)

        #self.dpm_local_prior = model_wrapper(
        #    local_prior,
        #    self.dpm_noise_schedule,
        #    model_type="noise",
        #    model_kwargs={"clip_feat": clip_feat, "condition_input": condition_input},
        #    # guidance_type="classifier-free",
        #    # unconditional_condition=None,
        #)

        #self.dpm_local_solver = DPM_Solver(
        #    self.dpm_local_prior,
        #    self.dpm_noise_schedule,
        #    algorithm_type="dpmsolver++"
        #)

        #x_noisy = self.dpm_local_solver.sample(
        #    x_noisy,
        #    steps=1000,
        #    order=3,
        #    skip_type="time_uniform",
        #    method="singlestep"
        #)
        ###################################


        sampled_list.append(x_noisy)
        output_dict['z_local'] = x_noisy

        # decode the latent (p_eps(x | h, z) = LION Decoder)
        output = self.vae.sample(num_samples=num_samples, decomposed_eps=sampled_list)
        if save_img:
            out_name = plot_points(output, "/tmp/tmp.png")
            print(f'INFO save plot image at {out_name}')
        output_dict['points'] = output
        return output_dict

    def get_mixing_component(self, noise_pred, t):
        # usage:
        # if global_prior.mixed_prediction:
        #     mixing_component = self.get_mixing_component(noise_pred, t)
        #     coeff = torch.sigmoid(global_prior.mixing_logit)
        #     noise_pred = (1 - coeff) * mixing_component + coeff * noise_pred

        alpha_bar = self.scheduler.alphas_cumprod[t]
        one_minus_alpha_bars_sqrt = np.sqrt(1.0 - alpha_bar)
        return noise_pred * one_minus_alpha_bars_sqrt
