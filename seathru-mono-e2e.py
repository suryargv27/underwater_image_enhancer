import os
import numpy as np
import PIL.Image as pil
import torch
from torchvision import transforms
import deps.monodepth2.networks as networks
from deps.monodepth2.utils import download_model_if_doesnt_exist
from skimage.restoration import denoise_tv_chambolle, estimate_sigma
from PIL import Image
import matplotlib.pyplot as plt
from seathru import *


image='/home/surya/IIITDM/Sem_6/DIP/under-water-enhancement/raw/10_img_.png'
output='output.png'
size=1024
monodepth_add_depth=2.0
monodepth_multiply_depth=10.0
model_name="mono_1024x320"


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

download_model_if_doesnt_exist(model_name)
model_path = os.path.join("models", model_name)
print("-> Loading model from ", model_path)
encoder_path = os.path.join(model_path, "encoder.pth")
depth_decoder_path = os.path.join(model_path, "depth.pth")

# LOADING PRETRAINED MODEL
print("   Loading pretrained encoder")
encoder = networks.ResnetEncoder(18, False)
loaded_dict_enc = torch.load(encoder_path, map_location=device)

# extract the height and width of image that this model was trained with
feed_height = loaded_dict_enc['height']
feed_width = loaded_dict_enc['width']
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)
encoder.to(device)
encoder.eval()

print("   Loading pretrained decoder")
depth_decoder = networks.DepthDecoder(
    num_ch_enc=encoder.num_ch_enc, scales=range(4))

loaded_dict = torch.load(depth_decoder_path, map_location=device)
depth_decoder.load_state_dict(loaded_dict)

depth_decoder.to(device)
depth_decoder.eval()

# Load image and preprocess
img = pil.open(image).convert('RGB')
img.thumbnail((size, size), Image.LANCZOS)
original_width, original_height = img.size
input_image = img.resize((feed_width, feed_height), pil.LANCZOS)
input_image = transforms.ToTensor()(input_image).unsqueeze(0)
print('Preprocessed image', flush=True)

# PREDICTION
input_image = input_image.to(device)
features = encoder(input_image)
outputs = depth_decoder(features)

disp = outputs[("disp", 0)]
disp_resized = torch.nn.functional.interpolate(
    disp, (original_height, original_width), mode="bilinear", align_corners=False)

# Saving colormapped depth image
disp_resized_np = disp_resized.squeeze().cpu().detach().numpy()
mapped_im_depths = ((disp_resized_np - np.min(disp_resized_np)) / (
        np.max(disp_resized_np) - np.min(disp_resized_np))).astype(np.float32)
print("Processed image", flush=True)
print('Loading image...', flush=True)
depths = preprocess_monodepth_depth_map(mapped_im_depths, monodepth_add_depth,
                                        monodepth_multiply_depth)

plt.imshow(depths)

recovered = run_pipeline(np.array(img) / 255.0, depths)
sigma_est = estimate_sigma(recovered, channel_axis=2, average_sigmas=True) / 10.0
recovered = denoise_tv_chambolle(recovered, sigma_est, channel_axis=2)
im = Image.fromarray((np.round(recovered * 255.0)).astype(np.uint8))
im.save(output, format='png')
print('Done.')
plt.show()