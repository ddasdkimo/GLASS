from glass import GLASS
import torch
from torchvision import transforms
import backbones
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def plot_heatmap(image, mask, alpha=0.5, colormap=cv2.COLORMAP_JET, savenm='heatmap.png'):
    """
    繪製熱區圖，將遮罩疊加在原始圖像上。
    :param image: 原始圖像 (PIL Image or numpy array)
    :param mask: 遮罩 (numpy array)
    :param alpha: 熱區圖的透明度 (0-1)
    :param colormap: OpenCV色彩映射
    :param savenm: 保存文件名
    """
    if isinstance(image, np.ndarray):
        img = image
    else:
        img = np.array(image)
    if mask.shape != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    if len(mask.shape) == 2:
        mask = cv2.applyColorMap((mask * 255).astype(np.uint8), colormap)
    heatmap = cv2.addWeighted(mask, alpha, img, 1 - alpha, 0)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    plt.savefig(savenm)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = backbones.load("wideresnet50")
model = GLASS(device)
model.load(
    backbone=backbone,
    layers_to_extract_from=('layer2', 'layer3'),
    device=device,
    input_shape=(3, 800, 800),
    pretrain_embed_dimension=1536,
    target_embed_dimension=1536,
    patchsize=3,
    meta_epochs=640,
    eval_epochs=1,
    dsc_layers=2,
    dsc_hidden=1024,
    dsc_margin=0.5,
    train_backbone=False,
    pre_proj=1
)
model_dir = "results/models/backbone_0"
dataset_name = "rai_gasket"
model.set_model_dir(model_dir, dataset_name)

image = Image.open('/dataset/rai/gasket/train/good/2244_13_13.png').convert('RGB')
model.loadermodel()
import time
start = time.time()
scores, masks = model.predict_image(image)
print('Time:', time.time()-start)
plot_heatmap(np.array(image), masks[0], alpha=0.5, savenm='heatmap.png')
