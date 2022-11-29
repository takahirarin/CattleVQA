import torch
import torch.nn as nn

class PatchEmbeddings(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self,config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size
        
        image_size = (image_size, image_size)
        patch_size = (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        # create patch with conv2d
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self,pixel_values):
        """ x here is image tokens"""
        batch_size, num_channels, height, weight = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError("Make sure that the channel dimension of the pixel values match with the one set in the configuration")
        x = self.projection(pixel_values)
        return x
