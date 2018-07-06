#from utils import one_hot
import os
#import numpy as np
import torch.utils.data
import PIL
import torch

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, x_transform=None, y_transform=None):
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.classes     = self.get_classes()
        self.samples     = self.get_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_src, y_src = self.samples[idx]
        x = self.get_x(x_src)
        y = self.get_y(y_src)
        if self.x_transform is not None: x = self.x_transform(x)
        if self.y_transform is not None: y = self.y_transform(y)
        return x,y

    def get_classes(self):
        """Return list of classes in a dataset."""
        raise NotImplementedError

    def get_samples(self):
        """Return list of tuples (x,y) that represents the datast"""
        raise NotImplementedError

    def get_x(self, x_src):
        """Return i-th example (image, wav, etc)."""
        raise NotImplementedError

    def get_y(self, y_src):
        """Return i-th label."""
        raise NotImplementedError


class SingleLabelClassificationDataset(BaseDataset):
    def get_y(self, y_src):
        return self.one_hot(y_src, len(self.classes))

    def one_hot(self, idx, num_classes):
        return torch.eye(num_classes)[idx]
        #return np.eye(num_classes)[idx]


class FolderDataset(BaseDataset):

    def __init__(self, root_dir, x_transform=None, y_transform=None, extensions=["png", "jpg", "jpeg"]):
        self.root_dir   = root_dir
        self.extensions = extensions
        super().__init__(x_transform, y_transform)

    def get_classes(self):
        classes = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        classes.sort()
        return classes

    def get_samples(self):
        samples = []
        
        for clss in self.classes:
            class_dir = os.path.join(self.root_dir, clss)

            for root, _, fnames in sorted(os.walk(class_dir)):
                for fname in sorted(fnames):
                    if self.has_allowed_extension(fname, self.extensions):
                        path = os.path.join(root, fname)
                        item = (path, self.classes.index(clss))
                        samples.append(item)

        return samples
    
    def has_allowed_extension(self, filename, extensions):
        filename_lower = filename.lower()
        return any(filename_lower.endswith(ext) for ext in extensions)


class ImageFolderDataset(FolderDataset, SingleLabelClassificationDataset):

    def __init__(self, root_dir, x_transform=None, y_transform=None, extensions=["png", "jpg", "jpeg"]):
        super().__init__(root_dir, x_transform, y_transform, extensions)

    def get_x(self, x_src):
        image    = PIL.Image.open(x_src)
        # image = np.array(image)
        return image
    
    #def resize_imgs(self):
    #def denorm(self):




#class CSVDataset(BaseDataset):
#    def get_samples(fn, skip_header=True, cat_separator = ' '):
#    """Parse filenames and label sets from a CSV file.
#
#    This method expects that the csv file at path :fn: has two columns. If it
#    has a header, :skip_header: should be set to True. The labels in the
#    label set are expected to be space separated.
#
#    Arguments:
#        fn: Path to a CSV file.
#        skip_header: A boolean flag indicating whether to skip the header.
#
#    Returns:
#        a two-tuple of (
#            image filenames,
#            a dictionary of filenames and corresponding labels
#        )
#    .
#    :param cat_separator: the separator for the categories column
#    """
#    df = pd.read_csv(fn, index_col=0, header=0 if skip_header else None, dtype=str)
#    fnames = df.index.values
#    df.iloc[:,0] = df.iloc[:,0].str.split(cat_separator)
#    return fnames, list(df.to_dict().values())[0]