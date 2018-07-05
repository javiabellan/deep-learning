


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.n         = self.get_n()
        self.c         = self.get_c()
        self.sz        = self.get_sz()

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x,y = self.get_x(idx),self.get_y(idx)
        return (x,y) if self.transform is None else tfm(x,y)

    @abstractmethod
    def get_n(self):
        """Return number of elements in the dataset == len(self)."""
        raise NotImplementedError

    @abstractmethod
    def get_c(self):
        """Return number of classes in a dataset."""
        raise NotImplementedError

    @abstractmethod
    def get_x(self, i):
        """Return i-th example (image, wav, etc)."""
        raise NotImplementedError

    @abstractmethod
    def get_y(self, i):
        """Return i-th label."""
        raise NotImplementedError



# SEE https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder

class FolderDataset():

    def __init__(self, root_dir, transform=None, target_transform=None):
        self.c       = self.find_classes(root_dir)
        self.samples = self.make_dataset(root_dir)
        self.n       = len(self.samples)

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        return classes

    def make_dataset(self, dir, extensions=["png", "jpg", "jpeg"]):
        samples = []
        
        for clss in self.c:
            class_dir = os.path.join(dir, clss)

            for root, _, fnames in sorted(os.walk(class_dir)):
                for fname in sorted(fnames):
                    if self.has_allowed_extension(fname, extensions):
                        path = os.path.join(root, fname)
                        item = (path, self.c.index(clss))
                        samples.append(item)

        return samples
    
    def has_allowed_extension(self, filename, extensions):
        filename_lower = filename.lower()
        return any(filename_lower.endswith(ext) for ext in extensions)





class ImageFolderDataset(BaseDataset):

    def __init__(self, csv_file, data_dir, transform=None):
        self.labels    = pd.read_csv(csv_file)
        self.data_dir  = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.labels.iloc[idx, 0])
        # image  = skimage.io.imread(img_name)   # Skimage
        image    = Image.open(img_name)
        # image = np.array(image)                # PIL image to numpy array
        
        label    = self.labels.iloc[idx, 1]
        sample   = {"image": image, "label": label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class FilesDataset(BaseDataset):
    def __init__(self, fnames, transform, path):
        self.path,self.fnames = path,fnames
        super().__init__(transform)
    def get_sz(self): return self.transform.sz
    def get_x(self, i): return open_image(os.path.join(self.path, self.fnames[i]))
    def get_n(self): return len(self.fnames)

    def resize_imgs(self, targ, new_path):
        dest = resize_imgs(self.fnames, targ, self.path, new_path)
        return self.__class__(self.fnames, self.y, self.transform, dest)

    def denorm(self,arr):
        """Reverse the normalization done to a batch of images.

        Arguments:
            arr: of shape/size (N,3,sz,sz)
        """
        if type(arr) is not np.ndarray: arr = to_np(arr)
        if len(arr.shape)==3: arr = arr[None]
        return self.transform.denorm(np.rollaxis(arr,1,4))

class CSVDataset(BaseDataset):
    def parse_csv_labels(fn, skip_header=True, cat_separator = ' '):
    """Parse filenames and label sets from a CSV file.

    This method expects that the csv file at path :fn: has two columns. If it
    has a header, :skip_header: should be set to True. The labels in the
    label set are expected to be space separated.

    Arguments:
        fn: Path to a CSV file.
        skip_header: A boolean flag indicating whether to skip the header.

    Returns:
        a two-tuple of (
            image filenames,
            a dictionary of filenames and corresponding labels
        )
    .
    :param cat_separator: the separator for the categories column
    """
    df = pd.read_csv(fn, index_col=0, header=0 if skip_header else None, dtype=str)
    fnames = df.index.values
    df.iloc[:,0] = df.iloc[:,0].str.split(cat_separator)
    return fnames, list(df.to_dict().values())[0]