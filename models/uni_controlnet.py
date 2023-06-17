import einops
import torch

from einops import rearrange, repeat
from torchvision.utils import make_grid

from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


class UniControlNet(LatentDiffusion):

    def __init__(self, mode, local_control_config=None, global_control_config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mode in ['local', 'global', 'uni']
        self.mode = mode
        if self.mode in ['local', 'uni']:
            self.local_adapter = instantiate_from_config(local_control_config)
            self.local_control_scales = [1.0] * 13
        if self.mode in ['global', 'uni']:
            self.global_adapter = instantiate_from_config(global_control_config)

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        
        if len(batch['local_conditions']) != 0:
            local_conditions = batch['local_conditions']
            if bs is not None:
                local_conditions = local_conditions[:bs]
            local_conditions = local_conditions.to(self.device)
            local_conditions = einops.rearrange(local_conditions, 'b h w c -> b c h w')
            local_conditions = local_conditions.to(memory_format=torch.contiguous_format).float()
        else:
            local_conditions = torch.zeros(1,1,1,1).to(self.device).to(memory_format=torch.contiguous_format).float()
        if len(batch['global_conditions']) != 0:
            global_conditions = batch['global_conditions']
            if bs is not None:
                global_conditions = global_conditions[:bs]
            global_conditions = global_conditions.to(self.device).to(memory_format=torch.contiguous_format).float()
        else:
            global_conditions = torch.zeros(1,1).to(self.device).to(memory_format=torch.contiguous_format).float()

        return x, dict(c_crossattn=[c], local_control=[local_conditions], global_control=[global_conditions])

    def apply_model(self, x_noisy, t, cond, global_strength=1, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if self.mode in ['global', 'uni']:
            assert cond['global_control'][0] != None
            global_control = self.global_adapter(cond['global_control'][0])
            cond_txt = torch.cat([cond_txt, global_strength*global_control], dim=1)
        if self.mode in ['local', 'uni']:
            assert cond['local_control'][0] != None
            local_control = torch.cat(cond['local_control'], 1)
            local_control = self.local_adapter(x=x_noisy, timesteps=t, context=cond_txt, local_conditions=local_control)
            local_control = [c * scale for c, scale in zip(local_control, self.local_control_scales)]
        
        if self.mode == 'global':
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt)
        else:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, local_control=local_control)
        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, plot_denoise_rows=False,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)

        c_cat = c["local_control"][0][:N]
        c_global = c["global_control"][0][:N]
        c = c["c_crossattn"][0][:N]
        
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["local_control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            samples, z_denoise_row = self.sample_log(cond={"local_control": [c_cat], "c_crossattn": [c], "global_control": [c_global]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat
            uc_global = torch.zeros_like(c_global)
            uc_full = {"local_control": [uc_cat], "c_crossattn": [uc_cross], "global_control": [uc_global]}
            samples_cfg, _ = self.sample_log(cond={"local_control": [c_cat], "c_crossattn": [c], "global_control": [c_global]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        if self.mode == 'global':
            h, w = 512, 512
        else:
            _, _, h, w = cond["local_control"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = []
        if self.mode in ['local', 'uni']:
            params += list(self.local_adapter.parameters())
        if self.mode in ['global', 'uni']:
            params += list(self.global_adapter.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            if self.mode in ['local', 'uni']:
                self.local_adapter = self.local_adapter.cuda()
            if self.mode in ['global', 'uni']:
                self.global_adapter = self.global_adapter.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            if self.mode in ['local', 'uni']:
                self.local_adapter = self.local_adapter.cpu()
            if self.mode in ['global', 'uni']:
                self.global_adapter = self.global_adapter.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
