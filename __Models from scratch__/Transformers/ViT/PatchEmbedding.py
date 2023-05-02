# https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # break-down the image in s1 x s2 patches and flat them
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size* patch_size * in_channels, emb_size)
        )
                
    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x


# https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=16, num_hiddens=512):
        super().__init__()

        self.conv = layers.Conv2D(
                      filters=num_hiddens,
                      kernel_size=patch_size,
                      stride=patch_size,
                      padding="valid", # "valid" means no padding
                      use_bias=False)

    def forward(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        return self.conv(X)
        x.flatten(2).transpose(1, 2) # BS #DIMS_PER_PATCH s1 s2 -> BS (s1 s2) #DIMS_PER_PATCH 



Which one do you prefer?

# INPUT SHAPE: [BS C H W] 
# H (image height) = nV (number of vertical Patches) * Hp (height of patch)
# W (image width)  = nH (number of horiz Patches)    * Wp (width of patch)
nn.EinOp("BS C (nV Hp) (nH Wp) -> BS (nV nH) (Hp Wp C)", Hp=patch_size, Wp=patch_size)
nn.Linear(in_dim=patch_size*patch_size*3, out_dim=emb_size)
# OUTPUT SHAPE: [BS, NUM_PATCHES, EMB_SIZE]

# INPUT:  BS C H W
nn.Conv2D(in_channels=3, out_channels=emb_size, kernel=patch_size, stride=patch_size, bias=false)
nn.EinOp("BS EMB_SIZE S1 S2 -> BS (S1 S2) EMB_SIZE") 
# OUTPUT: BS NUM_PATCHES EMB_SIZE


# INPUT:  BS C H W
# EINOP:  BS C (H S1) (W S2) -> BS (H W) (S1 S2 C)
# LINEAR: nn.Linear(patch_size * patch_size * in_channels, emb_size)
# OUTPUT: BS #PATCHES #DIMS_PER_PATCH

# INPUT:  BS C H W
# 2DCONV: nn.2dconv(in_channels=3, out_channels=emb_size, kernel&stride=patch_size, bias=false)
# EINOP:  BS #DIMS_PER_PATCH s1 s2 -> BS (s1 s2) #DIMS_PER_PATCH 
# OUTPUT: BS #PATCHES #DIMS_PER_PATCH
