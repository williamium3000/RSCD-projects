import mmcv
import os.path as osp
root = "data\LEVIR-CD-256\label"
dest = "data\LEVIR-CD-256\label"
crop_size=256

for img in mmcv.scandir(osp.join(root, "A"), suffix=".png", recursive=True):
    img = mmcv.imread(osp.join(osp.join(root, "A"), img))
    name, suffix = img.split(".")
    h, w, _ = img.shape
    for i in range(h // crop_size):
        for j in range(w // crop_size):
            mmcv.imwrite(img[i:i+crop_size, j:j+crop_size:], osp.join(osp.join(dest, "A", name + "_{}_{}.".format(i, j) + suffix)))
            
for img in mmcv.scandir(osp.join(root, "B"), suffix=".png", recursive=True):
    img = mmcv.imread(osp.join(osp.join(root, "A"), img))
    name, suffix = img.split(".")
    h, w, _ = img.shape
    for i in range(h // crop_size):
        for j in range(w // crop_size):
            mmcv.imwrite(img[i:i+crop_size, j:j+crop_size:], osp.join(osp.join(dest, "A", name + "_{}_{}.".format(i, j) + suffix)))

for img in mmcv.scandir(osp.join(root, "label"), suffix=".png", recursive=True):
    img = mmcv.imread(osp.join(osp.join(root, "A"), img))
    name, suffix = img.split(".")
    h, w, _ = img.shape
    for i in range(h // crop_size):
        for j in range(w // crop_size):
            mmcv.imwrite(img[i:i+crop_size, j:j+crop_size:], osp.join(osp.join(dest, "A", name + "_{}_{}.".format(i, j) + suffix)))
            