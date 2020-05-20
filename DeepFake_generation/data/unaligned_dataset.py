import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    and two directories: '/path/to/data/maskA' and '/path/to/data/maskB'  to load mask is needed
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.name = 'UnalignedDatasetMask'
        # get image path and size
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        self.A_paths = sorted(make_dataset(self.dir_A))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        # get transform parameter
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        self.face_mask = opt.face_mask
        # get mask path
        if self.face_mask:
            self.A_mask_dir = os.path.join(opt.dataroot, 'maskA')
            self.B_mask_dir = os.path.join(opt.dataroot, 'maskB')
            self.A_mask_paths = sorted(make_dataset(self.A_mask_dir))
            self.B_mask_paths = sorted(make_dataset(self.B_mask_dir))
            self.mask_transform = get_transform(self.opt, mask=True)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- image in the input domain
            B (tensor)       -- image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
            A_mask (tensor)  -- mask in the input domain
            B_mask (tensor)  -- mask in the target domain
        """

        random.seed(np.random.randint(2147483647))  # make a seed with numpy generator
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        index_B = random.randint(0, self.B_size - 1)
        # print(f'indexa {index % self.A_size}')
        B_path = self.B_paths[index_B] # get random (unpaired) B
        # print(f'indexb {index_B}')
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        # apply mask transformation
        if self.face_mask:
            A_mask_path = self.A_mask_paths[index % self.A_size]
            B_mask_path = self.B_mask_paths[index_B]
            A_mask = self.mask_transform(Image.open(A_mask_path).convert('RGB')) * (self.opt.face_weight - 1) + 1
            B_mask = self.mask_transform(Image.open(B_mask_path).convert('RGB')) * (self.opt.face_weight - 1) + 1
        else:
            A_mask = []
            B_mask = []
        # print(f'Amask: {A_mask_path}')
        # print(f'Aimage: {A_path}')
        # print(f'Bmask {B_mask_path}')
        # print(f'Bimage: {B_path}')
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_mask': A_mask, 'B_mask': B_mask}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

