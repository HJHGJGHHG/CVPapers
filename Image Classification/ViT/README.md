# Vision Transformer
> In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and ***a pure transformer*** applied directly to sequences of image patches can perform very well on image classification tasks. 

&emsp;&emsp;我 ViT 今天就是要给 Tranformer 正名！（指只用个 encoder 的屑 pure transformer ...）  

## 一、Method
&emsp;&emsp;直接上结构：  

<center><img src="1.png"  style="zoom:100%;" width="110%"/></center>

&emsp;&emsp;我们只需要理解一个核心的问题：NLP 处理的语言数据是序列化的，而CV 中处理的图像数据是三维的（height、width 和 channels）。如何将图像这种三维数据转化为序列化的数据？  
&emsp;&emsp;之前有文章直接将每个像素作为一个 token，显然会 TLE。本文处理方法是将图像被切割成一个个 patch，这些 patch 按照一定的顺序排列，就成了序列化的数据。这就是 ViT 的精髓所在。（你可能会说 就这？我的评价是细节非常多）  
&emsp;&emsp;详细而言：对于 $\mathbf{x}\in \mathbb{R}^{H\times W \times C}$ 的输入，转换为 $H\times W/P^2$ 个 $\mathbf{x}_p\in\mathbb{R}^{P^2\times C}$ patch，在实际代码中，通过一个卷积层和一个将 $P^2$ 两个维度展平的 flatten 实现：  
```
class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x
```
&emsp;&emsp;与同样只用 encoder 的 BERT 相同，加上 [CLS] 和 position embedding。（P.S：这里的位置编码为一维可训练的参数！！）文章中还分析了二维 PE 和组合策略，见原文 D.3 附录。  

## 二、Results and conclusion
&emsp;&emsp;效果自然是嘎嘎地，详细见原文，不多赘述了。文章 novelty 挺平庸的但是相关实验与分析很丰富，同时启发了后续 transformer-based 模型的发展，还展望了 CV 的预训练技术，算得上一篇相当扎实的经典文章。


