import albumentations as A

DEFAULT_AUGMENTATIONS = A.Compose([
    A.RandomRotate90(),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    #A.CoarseDropout(max_height=32, max_width=32, max_holes=3)
])
