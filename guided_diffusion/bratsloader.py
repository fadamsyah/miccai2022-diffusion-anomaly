import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
from scipy import ndimage

class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=False):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)

        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split('_')[3]
                    datapoint[seqtype] = os.path.join(root, f)
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            number=filedict['t1'].split('/')[4]
            nib_img = nibabel.load(filedict[seqtype])
            out.append(torch.tensor(nib_img.get_fdata()))
        out = torch.stack(out)
        out_dict = {}
        if self.test_flag:
            path2 = './data/brats/test_labels/' + str(
                number) + '-label.nii.gz'


            seg=nibabel.load(path2)
            seg=seg.get_fdata()
            image = torch.zeros(4, 256, 256)
            image[:, 8:-8, 8:-8] = out
            label = seg[None, ...]
            if seg.max() > 0:
                weak_label = 1
            else:
                weak_label = 0
            out_dict["y"]=weak_label
        else:
            image = torch.zeros(4,256,256)
            image[:,8:-8,8:-8]=out[:-1,...]		#pad to a size of (256,256)
            label = out[-1, ...][None, ...]
            if label.max()>0:
                weak_label=1
            else:
                weak_label=0
            out_dict["y"] = weak_label

        return (image, out_dict, weak_label, label, number )

    def __len__(self):
        return len(self.database)

import copy

class BRATSDatasetVolume(torch.utils.data.Dataset):
    def __init__(self, root, start_idx=None, end_idx=None, axial_dim=155):
        super().__init__()

        self.seqtypes = ['t1n', 't1c', 't2w', 't2f']
        # ['t1', 't1ce', 't2', 'flair']

        start_idx = start_idx if start_idx else 0
        end_idx = end_idx if end_idx else axial_dim

        assert start_idx >= 0
        assert end_idx <= axial_dim

        self.volumes = {}

        self.study_name = os.path.basename(os.path.normpath(root))
        for seqtype in self.seqtypes+['seg']:
            img_path = os.path.join(root, f'{self.study_name}-{seqtype}.nii.gz')
            img = nibabel.load(img_path).get_fdata()
            self.volumes[seqtype] = copy.deepcopy(img)
        
        self.list_idx = list(range(start_idx, end_idx))
    
    def __len__(self):
        return len(self.list_idx)

    def __getitem__(self, index):
        axial_index = self.list_idx[index]

        number = f'{self.study_name}-{axial_index}'
        
        out_dict = {}

        list_img = []
        for key, val in self.volumes.items():
            if key == 'seg':
                continue
            
            # Normalize to 0 - 1
            img_data = copy.deepcopy(val[..., axial_index])
            x = img_data - np.nanmin(img_data)
            y = np.nanmax(img_data) - np.nanmin(img_data)
            y = y if y != 0 else 1.0
            img_data = x / y  # (240, 240)

            # Append
            list_img.append(torch.tensor(img_data))

        image = torch.zeros(4, 256, 256)
        image[:, 8:-8, 8:-8] = torch.stack(list_img)

        seg = copy.deepcopy(self.volumes['seg'][..., axial_index])
        label = seg[None, ...]
        if seg.max() > 0:
            weak_label = 1
        else:
            weak_label = 0
        out_dict["y"]=weak_label
        
        return (image, out_dict, weak_label, label, number )


class BRATSDatasetFullVolume(torch.utils.data.Dataset):
    def __init__(self, root, skip_healthy_slices=True):
        super().__init__()
        
        self.seqtypes = ['t1n', 't1c', 't2w', 't2f']
        
        self.volumes = {}
        
        self.study_name = os.path.basename(os.path.normpath(root))
        for seqtype in self.seqtypes+['seg']:
            img_path = os.path.join(root, f'{self.study_name}-{seqtype}.nii.gz')
            img = nibabel.load(img_path).get_fdata()
            self.volumes[seqtype] = copy.deepcopy(img)
        
        if skip_healthy_slices == True:
            self.list_axial_indices = []
            for axial_idx in range(self.volumes['seg'].shape[-1]):
                seg_slice = self.volumes['seg'][..., axial_idx]
                seg_slice = np.where(seg_slice > 0, 1, 0)
                if seg_slice.max() > 0:
                    self.list_axial_indices.append(axial_idx)
        else:
            self.list_axial_indices = list(range(self.volumes['seg'].shape[-1]))
    
    def __len__(self):
        return len(self.list_axial_indices)
    
    def __getitem__(self, index):
        axial_index = self.list_axial_indices[index]
        
        list_img, list_vmin, list_vmax = [], [], []
        for key, val in self.volumes.items():
            if key == 'seg':
                continue
            
            # Normalize to 0 - 1
            img_data = copy.deepcopy(val[..., axial_index])
            vmin = np.nanmin(img_data)
            vmax = np.nanmax(img_data)
            
            y = vmax - vmin
            y = y if y != 0 else 1.0
            
            img_data = (img_data - vmin) / y  # (240, 240)
            
            # Append
            list_img.append(torch.tensor(img_data))
            list_vmin.append(vmin)
            list_vmax.append(vmax)
        
        image = torch.zeros(4, 256, 256)
        image[:, 8:-8, 8:-8] = torch.stack(list_img)
        
        if self.volumes['seg'][..., axial_index].max() > 0:
            weak_label = 1
        else:
            weak_label = 0
        
        return (image, weak_label, axial_index, list_vmin, list_vmax)