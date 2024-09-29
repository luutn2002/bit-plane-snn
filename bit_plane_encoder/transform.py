from torchvision.transforms import functional as F
from torch import nn
import torch
from torchvision import transforms
import numpy as np
from torchvision.transforms.functional import F_pil
import sys

try:
    import accimage
except ImportError:
    accimage = None

def to_tensor_no_div(pic):
    default_float_dtype = torch.get_default_dtype()

    if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()
        # backward compatibility
        if isinstance(img, torch.ByteTensor):
            return img.to(dtype=default_float_dtype)
        else:
            return img

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic).to(dtype=default_float_dtype)

    # handle PIL Image
    mode_to_nptype = {"I": np.int32, "I;16" if sys.byteorder == "little" else "I;16B": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))

    if pic.mode == "1":
        img = 255 * img
    img = img.view(pic.size[1], pic.size[0], F_pil.get_image_num_channels(pic))
    # put it from HWC to CHW format
    img = img.permute((2, 0, 1)).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.to(dtype=default_float_dtype)
    else:
        return img
    
class ToTensorNoDiv(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, image):
        return to_tensor_no_div(image)

class PILToTensor(nn.Module):
    def __init__(self, div=None):
        super().__init__()
        self.div = div
    
    def forward(self, image):
        image = image.convert(mode='RGB')
        image = F.pil_to_tensor(image)
        if self.div: image = image.float()/self.div
        return image
    
class ConstrainedResize(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        assert isinstance(size, int)
        self.size = size

    def forward(self, img):
        if img.shape[-1] >= img.shape[-2]: max_dim = img.shape[-1]
        else: max_dim = img.shape[-2]

        if max_dim > self.size: return transforms.Resize((self.size, self.size))(img)
        else: return transforms.Resize((max_dim, max_dim))(img)