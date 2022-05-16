import mmcv
import os.path as osp
import tqdm
root = "data/S2Looking/val"
dest = "data/S2Looking-256"
name_list_path = osp.join(dest, "val.txt")
crop_size=256
name_list = []
for _ in tqdm.tqdm(mmcv.scandir(osp.join(root, "A"), suffix=".png", recursive=True)):
    img = mmcv.imread(osp.join(osp.join(root, "A"), _))
    name, suffix = _.split(".")
    h, w, _ = img.shape
    for i in range(h // crop_size):
        for j in range(w // crop_size):
            dest_path = osp.join(osp.join(dest, "A", name + "_{}_{}.".format(i, j) + suffix))
            mmcv.mkdir_or_exist(osp.dirname(dest_path))
            mmcv.imwrite(img[i:i+crop_size, j:j+crop_size:], dest_path)
            name_list.append(osp.basename(dest_path))
for _ in tqdm.tqdm(mmcv.scandir(osp.join(root, "B"), suffix=".png", recursive=True)):
    img = mmcv.imread(osp.join(osp.join(root, "B"), _))
    name, suffix = _.split(".")
    h, w, _ = img.shape
    for i in range(h // crop_size):
        for j in range(w // crop_size):
            dest_path = osp.join(osp.join(dest, "B", name + "_{}_{}.".format(i, j) + suffix))
            mmcv.mkdir_or_exist(osp.dirname(dest_path))
            mmcv.imwrite(img[i:i+crop_size, j:j+crop_size:], dest_path)
            assert osp.basename(dest_path) in name_list

for _ in tqdm.tqdm(mmcv.scandir(osp.join(root, "label"), suffix=".png", recursive=True)):
    img = mmcv.imread(osp.join(osp.join(root, "label"), _))
    name, suffix = _.split(".")
    h, w, _ = img.shape
    for i in range(h // crop_size):
        for j in range(w // crop_size):
            dest_path = osp.join(osp.join(dest, "label", name + "_{}_{}.".format(i, j) + suffix))
            mmcv.mkdir_or_exist(osp.dirname(dest_path))
            mmcv.imwrite(img[i:i+crop_size, j:j+crop_size:], dest_path)
            assert osp.basename(dest_path) in name_list
with open(name_list_path, "w") as f:
    f.write("\n".join(name_list))