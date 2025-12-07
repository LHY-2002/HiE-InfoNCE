import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms
import os
import numpy as np
from PIL import Image

import cv2

################################################################################
#U1652DatasetTrain
class U1652DatasetTrain(Dataset):
    def __init__(self, root, transforms, prob_flip=0.5, names=['satellite', 'drone']):
        super().__init__()
        self.transforms_drone = transforms['drone']
        self.transforms_satellite = transforms['satellite']
        self.root = root
        self.names = names
        self.prob_flip = prob_flip

        dict_path = {}
        for name in names:
            dict_ = {}
            for cls_name in os.listdir(os.path.join(root, name)):
                img_list = os.listdir(os.path.join(root, name, cls_name))
                img_path_list = [os.path.join(root, name, cls_name, img) for img in img_list]
                dict_[cls_name] = img_path_list
            dict_path[name] = dict_

        cls_names = os.listdir(os.path.join(root, names[0]))
        cls_names.sort()
        map_dict = {i: cls_names[i] for i in range(len(cls_names))}

        self.cls_names = cls_names
        self.map_dict = map_dict
        self.dict_path = dict_path

    def __getitem__(self, index_pair):
        """
        index_pair = (cls_idx, drone_img_idx)
        """
        cls_idx, drone_img_idx = index_pair
        cls_name = self.map_dict[cls_idx]

        sat_path = self.dict_path["satellite"][cls_name][0]
        img_s = cv2.imread(sat_path)
        img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)

        drone_path = self.dict_path["drone"][cls_name][drone_img_idx]
        img_d = cv2.imread(drone_path)
        img_d = cv2.cvtColor(img_d, cv2.COLOR_BGR2RGB)

        if np.random.random() < self.prob_flip:
            img_s = cv2.flip(img_s, 1)
            img_d = cv2.flip(img_d, 1)

        # image transforms
        if self.transforms_satellite is not None:
            img_s = self.transforms_satellite(image=img_s)['image']
        if self.transforms_drone is not None:
            img_d = self.transforms_drone(image=img_d)['image']

        return img_s, img_d, cls_idx

    def __len__(self):
        return len(self.cls_names)


class Sampler_University(object):

    def __init__(self, data_source, batchsize=8, sample_num=4):
        self.data_source = data_source
        self.data_len = len(data_source)
        self.batchsize = batchsize
        self.sample_num = sample_num

    def __iter__(self):
        indices = np.arange(0, self.data_len)
        np.random.shuffle(indices)

        result = []
        for idx in indices:
            cls_name = self.data_source.map_dict[idx]
            drone_imgs = self.data_source.dict_path["drone"][cls_name]
            chosen = np.random.choice(len(drone_imgs),
                                      size=self.sample_num,
                                      replace=(len(drone_imgs) < self.sample_num))
            for c in chosen:
                result.append((idx, c))
        return iter(result)

    def __len__(self):
        return self.data_len * self.sample_num


def train_collate_fn(batch):
    img_s, img_d, ids = zip(*batch)
    ids = torch.tensor(ids, dtype=torch.int64)
    return [torch.stack(img_s, dim=0), ids], [torch.stack(img_d, dim=0), ids]




################################################################################
#U1652DatasetEval
def get_data(path):
    # drone:data = {'0839':{'path':'E:\\Univerity-Release\\train\\drone\\0839','files':['image-01.jpeg','image-02.jpeg',...]},'0842':{...}}
    data = {}
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            data[name] = {"path": os.path.join(root, name)}
            for _, _, files in os.walk(data[name]["path"], topdown=False):
                data[name]["files"] = files

    return data


class U1652DatasetEval(Dataset):

    def __init__(self, data_folder, mode, transforms=None, sample_ids=None, gallery_n=-1):
        super().__init__()

        self.data_dict = get_data(data_folder)

        # use only folders that exists for both gallery and query
        self.ids = list(self.data_dict.keys())

        self.transforms = transforms

        self.given_sample_ids = sample_ids

        self.images = []
        self.sample_ids = []

        self.mode = mode

        self.gallery_n = gallery_n

        for i, sample_id in enumerate(self.ids):

            for j, file in enumerate(self.data_dict[sample_id]["files"]):
                self.images.append("{}/{}".format(self.data_dict[sample_id]["path"],
                                                  file))

                self.sample_ids.append(sample_id)

    def __getitem__(self, index):

        img_path = self.images[index]
        sample_id = self.sample_ids[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # if self.mode == "sat":
        #
        #    img90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        #    img180 = cv2.rotate(img90, cv2.ROTATE_90_CLOCKWISE)
        #    img270 = cv2.rotate(img180, cv2.ROTATE_90_CLOCKWISE)
        #
        #    img_0_90 = np.concatenate([img, img90], axis=1)
        #    img_180_270 = np.concatenate([img180, img270], axis=1)
        #
        #    img = np.concatenate([img_0_90, img_180_270], axis=0)

        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        label = int(sample_id)
        if self.given_sample_ids is not None:
            if sample_id not in self.given_sample_ids:
                label = -1

        return img, label

    def __len__(self):
        return len(self.images)

    def get_sample_ids(self):
        return set(self.sample_ids)