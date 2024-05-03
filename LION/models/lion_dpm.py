from models.vae_adain import Model as VAE
from models.latent_points_ada_localprior import PVCNN2Prior as LocalPrior
from utils.diffusion_pvd import DiffusionDiscretized
from utils.vis_helper import plot_points
from utils.model_helper import import_model
from diffusers import DDPMScheduler
import torch
from matplotlib import pyplot as plt

from .dpm_solver_pytorch import *


class LionDPM(object):
    def __init__(self, cfg):
        print(cfg.ddpm)
        self.vae = VAE(cfg).cuda()
        GlobalPrior = import_model(cfg.latent_pts.style_prior)
        self.global_prior = GlobalPrior(cfg.sde, cfg.latent_pts.style_dim, cfg).cuda()
        
        # create a ddpm schedule to get the alpha_bars values for the noise schedule
        self.ddpm_scheduler = DDPMScheduler(clip_sample=False,
                                       beta_start=cfg.ddpm.beta_1, beta_end=cfg.ddpm.beta_T, beta_schedule=cfg.ddpm.sched_mode,
                                       num_train_timesteps=cfg.ddpm.num_steps, variance_type=cfg.ddpm.model_var_type)
        
        # create the noise schedule
        self.scheduler = NoiseScheduleVP(schedule='discrete', alphas_cumprod=self.ddpm_scheduler.alphas_cumprod)
        
        # wrap the global prior to use DPM-Solver
        self.global_fn = model_wrapper(
            self.global_prior,
            self.scheduler,
            model_type='noise', # "noise" or "x_start" or "v" or "score"
            # model_kwargs=model_kwargs,
        )
        self.global_dpm_solver = DPM_Solver(self.global_fn, self.scheduler, algorithm_type="dpmsolver")
        
        self.local_prior = LocalPrior(cfg.sde, cfg.shapelatent.latent_dim, cfg).cuda()
        
        self.priors = torch.nn.ModuleList([self.global_prior, self.local_prior])
        self.diffusion = DiffusionDiscretized(None, None, cfg)
        self.dpm_steps = cfg.ddpm.dpm_step

        self.use_dpm = True
        # self.load_model(cfg)

    def load_model(self, model_path):
        # model_path = cfg.ckpt.path
        ckpt = torch.load(model_path)
        self.priors.load_state_dict(ckpt['dae_state_dict'])
        self.vae.load_state_dict(ckpt['vae_state_dict'])
        print(f'INFO finish loading from {model_path}')

    @torch.no_grad()
    def sample(self, num_samples=1, clip_feat=None, save_img=False):
        self.ddpm_scheduler.set_timesteps(1000, device='cuda')
        timesteps = self.ddpm_scheduler.timesteps

        latent_shape = self.vae.latent_shape()
        global_prior, local_prior = self.priors[0], self.priors[1]
        assert(not local_prior.mixed_prediction and not global_prior.mixed_prediction)
        sampled_list = []
        output_dict = {}

        # start sample global prior
        x_T_shape = [num_samples] + latent_shape[0]
        x_noisy = torch.randn(size=x_T_shape, device='cuda')
        condition_input = None

        # Initial for loop for sampling in LION Code
        if not self.use_dpm:
            for i, t in enumerate(timesteps):
                t_tensor = torch.ones(num_samples, dtype=torch.int64, device='cuda') * (t+1)
                noise_pred = self.global_prior(x=x_noisy, t=t_tensor.float(),
                    condition_input=condition_input, clip_feat=clip_feat)
                x_noisy = self.ddpm_scheduler.step(noise_pred, t, x_noisy).prev_sample
        else:
            # New method for unconditional sampling using DPM-Solver
            # start sample global prior (p_theta(z) = distribution of global shape latent DDM)
            print(f"Performing {self.dpm_steps} diffusion steps")
            x_noisy = self.global_dpm_solver.sample(x=x_noisy, steps=self.dpm_steps)

        sampled_list.append(x_noisy)
        output_dict['z_global'] = x_noisy

        # Set up to condition on global sample to generate local 
        condition_input = x_noisy
        condition_input = self.vae.global2style(condition_input)

        # start sample local prior (p_phi(h | z) = DDM modeling the piont cloud-structured latents)
        x_T_shape = [num_samples] + latent_shape[1]
        x_noisy = torch.randn(size=x_T_shape, device='cuda')

        # Initial for-loop for sampling in LION code
        if not self.use_dpm:
            for i, t in enumerate(timesteps):
                t_tensor = torch.ones(num_samples, dtype=torch.int64, device='cuda') * (t+1)
                noise_pred = self.local_prior(x=x_noisy, t=t_tensor.float(),
                    condition_input=condition_input, clip_feat=clip_feat)
                x_noisy = self.ddpm_scheduler.step(noise_pred, t, x_noisy).prev_sample
        else:
            local_prior_kwargs = {'condition_input': condition_input}
            guidance_scale = 0.5 # TODO: Tune
            local_fn = model_wrapper(
                self.local_prior,
                self.scheduler,
                model_type="noise",  # or "x_start" or "v" or "score"
                model_kwargs=local_prior_kwargs,
                guidance_type="classifier-free",
                condition=condition_input,
                unconditional_condition=torch.ones(condition_input.shape, device=condition_input.device), # TODO: What is unconditional_input supposed to be
                guidance_scale=guidance_scale,
            )
            local_dpm_solver = DPM_Solver(local_fn, self.scheduler, algorithm_type="dpmsolver++") #"dpmsolver++" recommended
        
            print(f"Performing {self.dpm_steps} diffusion steps")
            x_noisy = local_dpm_solver.sample(
                x_noisy,
                steps=self.dpm_steps,
                order=2,
                skip_type="time_uniform",
                method="multistep", #"multistep" recommended
            )

        sampled_list.append(x_noisy)
        output_dict['z_local'] = x_noisy

        # decode the latent
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
