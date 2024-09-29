# bit-plane-snn
The official repository of "Improving Spiking Neural Network Accuracy With Color Model Information Encoded Bit Planes", contain the mentioned hybrid encoder.

[Paper]()

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
        PILToTensor(),
        torchvision.transforms.Resize((224, 224)),
    ])

tensor = transform(img)
encoder = MixedEncoder(T = 10, #Time step of rate coding 
                    color_model ="rgb") #Selected color model information to be used, default to rgb.
encoded_tensor = encoder(tensor)
```

### Step 2: Environment setup and repo download

## Acknowledgement

Special thanks to [SpikingJelly](https://github.com/fangwei123456/spikingjelly) for their guides to SNN and [Kornia](https://github.com/kornia/kornia) for color conversion algorithm reference. 

## Citation

```bibtex

```
