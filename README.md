# bit-plane-snn
*Notes: work accepted for publication at IEEE Access*

The official repository of "Improvement of Spiking Neural Network with Bit Planes and Color Models", contain the mentioned hybrid encoder.

[Paper](https://ieeexplore.ieee.org/document/11261679)

## Quickstart

This is a quickstart guide on how to use our encoder as a package 

### Step 1: Environment setup and repo download

To setup the environment testing with this encoder, you will need Pytorch and SpikingJelly. We suggest using conda environment with:

```bash
$ conda create -n env python=3.12.2
$ conda install pytorch=2.3.0 torchvision=0.18.0 torchaudio=2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia #As latest pytorch conda guide, change cuda version suitable to your case.
$ pip install spikingjelly
$ pip install git+https://github.com/luutn2002/bit-plane-snn.git
```

or clone and modify locally:

```bash
$ git clone https://github.com/luutn2002/bit-plane-snn.git
```

### Step 2: Import and usage

To use the encoder, we can simply import and encode with:

```python
from bit_plane_encoder.transform import PILToTensor
from bit_plane_encoder import MixedEncoder #Bit plane encode with rate coding
from PIL import Image
import requests
import torchvision

#Load image to PIL image
img = Image.open(requests.get("https://dummyimage.com/500x400/ff6699/000", stream=True).raw)

#Transform PIL images to tensor
transform = torchvision.transforms.Compose([
        PILToTensor(), #Conver PIL to unnormalized tensor.
        torchvision.transforms.Resize((224, 224)),
    ])

tensor = transform(img)
encoder = MixedEncoder(T = 10, #Time step of rate coding 
                    color_model ="rgb") #Selected color model information to be used, default to rgb.
encoded_tensor = encoder(tensor)
```

To only use the bit plane encoder, you can use:

```python
from bit_plane_encoder.transform import PILToTensor
from bit_plane_encoder import BitplaneColorEncoder #Bit plane encode only
from PIL import Image
import requests
import torchvision

#Load image to PIL image
img = Image.open(requests.get("https://dummyimage.com/500x400/ff6699/000", stream=True).raw)

#Transform PIL images to tensor
transform = torchvision.transforms.Compose([
        PILToTensor(), #Conver PIL to unnormalized tensor.
        torchvision.transforms.Resize((224, 224)),
    ])

tensor = transform(img)
encoder = BitplaneColorEncoder(color_model ="rgb") #Selected color model information to be used, default to rgb.
encoded_tensor = encoder(tensor)
```
Current supported color models include: RGB, CMY, YCbCr, HSL, HSV, LAB, XYZ
```python
color_models = ["rgb", "cmy", "ycbcr", "hsl", "hsv", "lab", "xyz"]
encoder = BitplaneColorEncoder(color_model = color_models[0])
```
## Acknowledgement

Special thanks to [SpikingJelly](https://github.com/fangwei123456/spikingjelly) for their guides to SNN and [Kornia](https://github.com/kornia/kornia) for color conversion algorithm reference. 

## Citation

```bibtex
@ARTICLE{11261679,
  author={Luu, Nhan T. and Luu, Duong T. and Nam, Pham Ngoc and Thang, Truong Cong},
  journal={IEEE Access}, 
  title={Improvement of Spiking Neural Network With Bit Planes And Color Models}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Neurons;Image color analysis;Training;Spiking neural networks;Computational modeling;Tensors;Encoding;Biological system modeling;Deep learning;Membrane potentials;Spiking neural network;spike coding;image classification;bit planes;color models},
  doi={10.1109/ACCESS.2025.3635297}}
```
