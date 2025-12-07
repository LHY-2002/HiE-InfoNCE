from datasets.U1652Dataset import Sampler_University,U1652DatasetTrain,train_collate_fn
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(img_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

    val_transforms = A.Compose([
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    train_sat_transforms = A.Compose([
        A.ImageCompression(quality_range=(90, 100), p=0.5),
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.3, hue=0.3, p=0.5),
        A.OneOf([A.AdvancedBlur(p=1.0), A.Sharpen(p=1.0),], p=0.3),
        A.OneOf([A.GridDropout(ratio=0.2, p=1.0),
                            A.CoarseDropout(num_holes_range=(3, 5),
                                            hole_height_range=(int(0.1 * img_size[0]), int(0.2 * img_size[0])),
                                            hole_width_range=(int(0.1 * img_size[1]), int(0.2 * img_size[1])),
                                            fill=0,
                                            p=1.0),
                            ], p=0.3),
        A.RandomRotate90(p=1.0),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    train_drone_transforms = A.Compose([
        A.ImageCompression(quality_range=(90, 100), p=0.5),
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.ColorJitter(brightness=0.15, contrast=0.7, saturation=0.3, hue=0.3, p=0.5),
        A.OneOf([A.AdvancedBlur(p=1.0), A.Sharpen(p=1.0),], p=0.3),
        A.OneOf([A.GridDropout(ratio=0.2, p=1.0),
                            A.CoarseDropout(num_holes_range=(3, 5),
                                            hole_height_range=(int(0.1 * img_size[0]), int(0.2 * img_size[0])),
                                            hole_width_range=(int(0.1 * img_size[1]), int(0.2 * img_size[1])),
                                            fill=0,
                                            p=1.0),
                            ], p=0.3),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    data_transforms = {
        'satellite': train_sat_transforms,
        'drone': train_drone_transforms,
        'val': val_transforms,
    }

    return data_transforms




def get_train_dataloader(config, data_transforms):
    image_datasets = U1652DatasetTrain(config.data_folder_train, transforms=data_transforms, prob_flip=config.prob_flip)
    samper = Sampler_University(image_datasets, batchsize=config.batch_size, sample_num=config.sample_num)
    dataloaders =torch.utils.data.DataLoader(image_datasets,
                                             batch_size=config.batch_size,
                                             drop_last=True,
                                             sampler=samper,
                                             num_workers=config.num_workers,
                                             pin_memory=True,
                                             collate_fn=train_collate_fn)
    dataset_sizes = {x: len(image_datasets)*config.sample_num for x in ['satellite', 'drone']}
    class_names = image_datasets.cls_names
    return dataloaders, class_names, dataset_sizes

