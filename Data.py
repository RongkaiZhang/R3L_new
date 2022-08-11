import glob
import numpy as np
import PIL.Image as pil_image
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms

class Dataset(object):
    def __init__(self, images_dir, patch_size):
        self.image_files = sorted(glob.glob(images_dir + '/*'))
        #self.label_files = sorted(glob.glob(label_dir + '/*'))
        self.patch_size = patch_size
        
        

    def __getitem__(self, idx):
        image = pil_image.open(self.image_files[idx])

        transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomRotation(80),
                                        transforms.RandomCrop(self.patch_size)
                                        ])
        image = transform(image)
        image = np.array(image).astype(np.float32)

        return image

    def __len__(self):
        return len(self.image_files)

if __name__ == '__main__':
    dataset = Dataset('test/',70)
    data = DataLoader(dataset=dataset,
                      batch_size=2,
                      shuffle=True,
                      num_workers=1,
                      pin_memory=True,
                      drop_last=True)

    for inputs in data:
        im = np.array(inputs).astype(np.float32)/255
