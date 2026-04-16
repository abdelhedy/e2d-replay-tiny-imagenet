# Teacher checkpoints

The pipeline expects two Tiny-ImageNet ResNet checkpoints here:

```
checkpoints/tinyimagenet/rn18_200ep.pth   # ResNet-18, ~90 MB
checkpoints/tinyimagenet/rn50_200ep.pth   # ResNet-50, ~190 MB
```

They are torchvision ResNets adapted to 64×64 inputs (3×3 `conv1`, no max-pool, 200-way `fc`), trained for 200 epochs on Tiny-ImageNet. They come from [zeyuanyin/tiny-imagenet](https://github.com/zeyuanyin/tiny-imagenet).

The files are git-ignored because they exceed GitHub's 100 MB file limit. Download them and place them in this folder, or pass other paths with `--teacher-checkpoints`.
