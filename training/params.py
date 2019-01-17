from model.u_net import get_unet_128, get_unet_256, get_unet_512, get_unet_1024

input_size = 256

max_epochs =10
batch_size = 20

orig_width = 256
orig_height = 256

threshold = 0.5

model_factory = get_unet_256
