import sys
import torch
from torch import nn
import math
from spikingjelly.activation_based import encoding

class BitplaneColorEncoder(nn.Module):
    def __init__(self,
                 color_model:str="rgb"):
        super().__init__()
        self.color_model = color_model

    def _rgb_to_y(self, r: torch.Tensor, g: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        y = 0.299 * r + 0.587 * g + 0.114 * b
        return y


    def rgb_to_ycbcr(self, image: torch.Tensor, scale=False) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

        if len(image.shape) < 3 or image.shape[-3] != 3:
            raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

        r = image[..., 0, :, :]
        g = image[..., 1, :, :]
        b = image[..., 2, :, :]

        delta: float = 0.5
        y = self._rgb_to_y(r, g, b)
        cb = (b - y) * 0.564 + delta
        cr = (r - y) * 0.713 + delta
        if scale: return torch.stack([y*235, cb*240, cr*240], -3).byte()
        else: return torch.stack([y, cb, cr], -3)

    def rgb_to_xyz(self, image: torch.Tensor, scale=False) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

        if len(image.shape) < 3 or image.shape[-3] != 3:
            raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

        r = image[..., 0, :, :]
        g = image[..., 1, :, :]
        b = image[..., 2, :, :]

        x = 0.412453 * r + 0.357580 * g + 0.180423 * b
        y = 0.212671 * r + 0.715160 * g + 0.072169 * b
        z = 0.019334 * r + 0.119193 * g + 0.950227 * b

        if scale: out = torch.stack([x*95.047, y*100, z*108.883], -3).byte()
        else: out = torch.stack([x, y, z], -3)
        return out
    
    def rgb_to_cmy(self, image: torch.Tensor, scale=False) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

        if len(image.shape) < 3 or image.shape[-3] != 3:
            raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

        r = image[..., 0, :, :]
        g = image[..., 1, :, :]
        b = image[..., 2, :, :]

        c = 1 - r
        m = 1 - g
        y = 1 - b

        if scale: out = torch.stack([c*100, m*100, y*100], -3).byte()
        else: out = torch.stack([c, m, y], -3)
        return out
    
    def rgb_to_linear_rgb(self, image: torch.Tensor) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

        if len(image.shape) < 3 or image.shape[-3] != 3:
            raise ValueError(f"Input size must have a shape of (*, 3, H, W).Got {image.shape}")

        lin_rgb = torch.where(image > 0.04045, torch.pow(((image + 0.055) / 1.055), 2.4), image / 12.92)

        return lin_rgb

    def rgb2hsl_torch(self, rgb: torch.Tensor) -> torch.Tensor:
        cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
        cmin = torch.min(rgb, dim=1, keepdim=True)[0]
        delta = cmax - cmin
        hsl_h = torch.empty_like(rgb[:, 0:1, :, :])
        cmax_idx[delta == 0] = 3
        hsl_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
        hsl_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
        hsl_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
        hsl_h[cmax_idx == 3] = 0.
        hsl_h /= 6.

        hsl_l = (cmax + cmin) / 2.
        hsl_s = torch.empty_like(hsl_h)
        hsl_s[hsl_l == 0] = 0
        hsl_s[hsl_l == 1] = 0
        hsl_l_ma = torch.bitwise_and(hsl_l > 0, hsl_l < 1)
        hsl_l_s0_5 = torch.bitwise_and(hsl_l_ma, hsl_l <= 0.5)
        hsl_l_l0_5 = torch.bitwise_and(hsl_l_ma, hsl_l > 0.5)
        hsl_s[hsl_l_s0_5] = ((cmax - cmin) / (hsl_l * 2.))[hsl_l_s0_5]
        hsl_s[hsl_l_l0_5] = ((cmax - cmin) / (- hsl_l * 2. + 2.))[hsl_l_l0_5]
        return torch.cat([hsl_h*179.0, hsl_s*255.0, hsl_l*255.0], dim=1).byte()


    def rgb2hsv_torch(self, rgb: torch.Tensor) -> torch.Tensor:
        cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
        cmin = torch.min(rgb, dim=1, keepdim=True)[0]
        delta = cmax - cmin
        hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
        cmax_idx[delta == 0] = 3
        hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
        hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
        hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
        hsv_h[cmax_idx == 3] = 0.
        hsv_h /= 6.
        hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
        hsv_v = cmax
        return torch.cat([hsv_h*359.0, hsv_s*100.0, hsv_v*100.0], dim=1).byte()
    
    def rgb_to_lab(self, image: torch.Tensor) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

        if len(image.shape) < 3 or image.shape[-3] != 3:
            raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

        # Convert from sRGB to Linear RGB
        lin_rgb = self.rgb_to_linear_rgb(image)

        xyz_im = self.rgb_to_xyz(lin_rgb)

        # normalize for D65 white point
        xyz_ref_white = torch.tensor([0.95047, 1.0, 1.08883], device=xyz_im.device, dtype=xyz_im.dtype)[..., :, None, None]
        xyz_normalized = torch.div(xyz_im, xyz_ref_white)

        threshold = 0.008856
        power = torch.pow(xyz_normalized.clamp(min=threshold), 1 / 3.0)
        scale = 7.787 * xyz_normalized + 4.0 / 29.0
        xyz_int = torch.where(xyz_normalized > threshold, power, scale)

        x = xyz_int[..., 0, :, :]
        y = xyz_int[..., 1, :, :]
        z = xyz_int[..., 2, :, :]

        L = (116.0 * y) - 16.0
        a = 500.0 * (x - y)
        _b = 200.0 * (y - z)

        out = torch.stack([L, a, _b], dim=-3).byte()

        return out
    
    def binarize(self, x, max=255):
        bit_per_channels = math.ceil(math.log2(max))
        res = []
        for _ in range(bit_per_channels):
            res.append(torch.remainder(x, 2))
            x = torch.div(x, 2, rounding_mode="floor")
        return torch.stack(res)

    def forward(self, x):
        if sys.hexversion >= 0x030a00f0:
            match self.color_model:
                case "rgb": return self.binarize(x).float()
                    
                case "hsl":
                    normalized = torch.div(x, 255.0)
                    hsl = self.rgb2hsl_torch(normalized)
                    return self.binarize(hsl, max=359).float()

                case "hsv":
                    normalized = torch.div(x, 255.0)
                    hsv = self.rgb2hsv_torch(normalized)
                    return self.binarize(hsv).float()
                
                case "cmy":
                    normalized = torch.div(x, 255.0)
                    cmy = self.rgb_to_cmy(normalized)
                    return self.binarize(cmy).float()
                
                case "xyz":
                    normalized = torch.div(x, 255.0)
                    xyz = self.rgb_to_xyz(normalized, scale = True)
                    return self.binarize(xyz, max=109).float()
                
                case "lab":
                    normalized = torch.div(x, 255.0)
                    lab = self.rgb_to_lab(normalized)
                    return self.binarize(lab, max=128).float()
                
                case "ycbcr":
                    normalized = torch.div(x, 255.0)
                    lab = self.rgb_to_ycbcr(normalized, scale=True)
                    return self.binarize(lab, max=240).float()
                
                case _: raise RuntimeError("Mode does not exist in encoder")
        else:
            if self.color_model == "rgb": return self.binarize(x).float()
                    
            elif self.color_model ==  "hsl":
                normalized = torch.div(x, 255.0)
                hsl = self.rgb2hsl_torch(normalized)
                return self.binarize(hsl, max=359).float()

            elif self.color_model ==  "hsv":
                normalized = torch.div(x, 255.0)
                hsv = self.rgb2hsv_torch(normalized)
                return self.binarize(hsv).float()
            
            elif self.color_model == "cmy":
                normalized = torch.div(x, 255.0)
                cmy = self.rgb_to_cmy(normalized)
                return self.binarize(cmy).float()
            
            elif self.color_model == "xyz":
                normalized = torch.div(x, 255.0)
                xyz = self.rgb_to_xyz(normalized, scale = True)
                return self.binarize(xyz, max=109).float()
            
            elif self.color_model == "lab":
                normalized = torch.div(x, 255.0)
                lab = self.rgb_to_lab(normalized)
                return self.binarize(lab, max=128).float()
            
            elif self.color_model == "ycbcr":
                normalized = torch.div(x, 255.0)
                lab = self.rgb_to_ycbcr(normalized, scale=True)
                return self.binarize(lab, max=240).float()
            
            else: raise RuntimeError("Mode does not exist in encoder")


class MixedEncoder(nn.Module):
    def __init__(self,
                 T:int = 10,
                 color_model:str="rgb"):
        super().__init__()
        self.T = T
        self.encoder = encoding.PoissonEncoder()
        self.color_encoder = BitplaneColorEncoder(color_model)
        
    def forward(self, x):
        normalized = torch.div(x, 255.0)
        encoded = torch.stack([self.encoder(normalized) for _ in range(self.T)])
        color_encoded = self.color_encoder(x)
        res = torch.cat([encoded, color_encoded], dim=0)
        return res