import torch
from torch.nn import Module
import numpy as np
from .config import DataAugmentorHParams

class DataAugmentor(Module):
    def __init__(self, hparams: DataAugmentorHParams, dim: int):
        super().__init__()
        self.hparams = hparams
        if self.hparams.rotation and not dim == 2:
            raise NotImplementedError("Rotation is only implemented for 2D data.")

    def forward(self, x, c):
        if self.hparams.rotation:
            rotation_range = self.hparams.rotation
            if rotation_range == True:
                rotation_range = [0, 2*np.pi]
            assert len(rotation_range) == 2, "rotation_range must be a list of two values"
            angle = torch.rand(x.shape[0], device=x.device)*(rotation_range[1] - rotation_range[0]) + rotation_range[0]
            x = self.rotate(x, angle)
        if self.hparams.translation:
            translation_range = self.hparams.translation
            if translation_range == True:
                translation_range = [-0.5, 0.5]
            assert len(translation_range) == 2, "translation_range must be a list of two values"
            translation = torch.rand_like(x)*(translation_range[1] - translation_range[0]) + translation_range[0]
            x = self.translate(x, translation)
        if self.hparams.flip:
            flip_chance = self.hparams.flip
            if flip_chance == True:
                flip_chance = 0.5
            x = self.flip(x, flip_chance)
        if self.hparams.add_noise:
            noise_level = self.hparams.add_noise
            if noise_level == True:
                noise_level = 0.1
            x = self.add_noise(x, noise_level)
        if self.hparams.scale:
            scale_range = self.hparams.scale
            if scale_range == True:
                scale_range = [0.5, 2.0]
            assert len(scale_range) == 2, "scale_range must be a list of two values"
            scale = torch.rand(x.shape[0], device=x.device)*(scale_range[1] - scale_range[0]) + scale_range[0]
            x = self.scale(x, scale)
        return x, c
    
    def rotate(self, x, angle):
        angle = angle.view(-1, 1)
        rotation_matrix = torch.cat([torch.cos(angle), -torch.sin(angle), torch.sin(angle), torch.cos(angle)], dim=1).view(-1, 2, 2)
        reduce = False
        if x.ndim == 2:
            x = x.view(-1, 1, 2)
            reduce = True
        x = torch.bmm(x, rotation_matrix)
        if reduce:
            x = x.view(-1, 2)
        return x
    
    def translate(self, x, translation):
        return x + translation
    
    def flip(self, x, flip_chance):
        if x.ndim == 2:
            flip_mask = torch.rand_like(x) < flip_chance
        elif x.ndim == 3:
            flip_mask = torch.rand_like(x[:, 0, :]) < flip_chance
            flip_mask = flip_mask.unsqueeze(1).repeat(1, x.shape[1], 1)
        else:
            raise NotImplementedError(f"Received input with ndim == {x.ndim}, but flip is only implemented for ndim == 2 or ndim == 3")
        return torch.where(flip_mask, -x, x)

    def add_noise(self, x, noise_level):
        return x + torch.randn_like(x)*noise_level

    def scale(self, x, scale):
        return x*scale
        