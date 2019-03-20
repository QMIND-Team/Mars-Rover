from model.u_net import get_unet_128, get_unet_256, get_unet_512, get_unet_1024

input_size = 128

max_epochs =5
batch_size = 43

orig_width = 128
orig_height = 128

threshold = 0.5

model_factory = get_unet_128
