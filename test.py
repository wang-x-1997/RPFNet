#coding= utf_8
from __future__ import print_function
# from scipy.misc import imread, imresize
import argparse
import os
import cv2
from PIL import Image
import time
import glob
from net_utilis import *

def RGB2YCbCr(img):
    img = img * 255.0
    r, g, b = torch.split(img, 1, dim=1)

    y = 0.257 * r + 0.504 * g + 0.098 * b + 16
    y = y / 255.0

    cb = -0.148 * r - 0.291 * g + 0.439 * b + 128
    cb = cb / 255.0

    cr = 0.439 * r - 0.368 * g - 0.071 * b + 128
    cr = cr / 255.0

    img = torch.cat([y, cb, cr], dim=1)
    return img


def YCbCr2RGB(img,img_Y):
    img = RGB2YCbCr(img)*255
    y, cb, cr = torch.split(img, 1, dim=1)

    r = 1.164 * (img_Y*255 - 16) + 1.596 * (cr - 128)
    r = r / 255.0
    g = 1.164 * (img_Y*255 - 16) - 0.392 * (cb - 128) - 0.813 * (cr - 128)
    g = g / 255.0
    b = 1.164 * (img_Y*255 - 16) + 2.017 * (cb - 128)
    b = b / 255.0

    img = torch.cat([b, g, r], dim=1)
    return img*255


def prepare_data(dataset):
    # data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data_dir =dataset
    data = glob.glob(os.path.join(data_dir, "IR*.jpg"))
    data.extend(glob.glob(os.path.join(data_dir, "IR*.png")))
    data.extend(glob.glob(os.path.join(data_dir, "IR*.bmp")))
    a = data[0][len(str(data_dir))+1:-6]
    data.sort(key=lambda x:int(x[len(str(data_dir))+2:-4]))
    return data
def prepare_data1(dataset):
    # data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data_dir = dataset
    data = glob.glob(os.path.join(data_dir, "VIS*.jpg"))
    data.extend(glob.glob(os.path.join(data_dir, "VIS*.png")))
    data.extend(glob.glob(os.path.join(data_dir, "VIS*.bmp")))
    data.sort(key=lambda x:int(x[len(str(data_dir))+3:-4]))
    return data

def change(out):
    out1 = out.cpu()
    out_img = out1.data[0]
    # out_img = out_img.squeeze()
    out_img = out_img.numpy()
    out_img = out_img.transpose(1, 2, 0)
    return out_img
def change_gray(out):
    out1 = out.cpu()
    out_img = out1.data[0]
    out_img = out_img.squeeze()
    out_img = out_img.numpy()
    # out_img = out_img.transpose(1, 2, 0)
    return out_img
def load_image(x):
  imgA = Image.open(x)
  imgA = imgA.convert('L')
  imgA = np.asarray(imgA)
  imgA = np.atleast_3d(imgA).transpose(2, 0, 1).astype(np.float)
  imgA = torch.from_numpy(imgA).float()
  imgA = imgA.unsqueeze(0)
  return imgA

def load_rgb(x):
  imgA = Image.open(x).convert('RGB')
  # imgA = imgA.convert('L')
  imgA = np.asarray(imgA)
  imgA = np.atleast_3d(imgA).transpose(2, 0, 1).astype(np.float)
  imgA = torch.from_numpy(imgA).float()
  imgA = imgA.unsqueeze(0)
  return imgA
def RGB2Y(img):

    r, g, b = torch.split(img, 1, dim=1)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y


def norm_1(x):
    max1 = torch.max(x)
    min1 = torch.min(x)
    return (x-min1)/(max1-min1 + 1e-10)

def name_list(path):
    filename_list = os.listdir(path)
    source_name = []
    save_name = []
    for i in filename_list:
        filename = str(i)
        b = filename.split('.')
        source_name.append(filename)
        save_name.append(b[0])
    return source_name,save_name

start=time.time()
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()
device = torch.device("cuda")

p='5'
model =r'./Trained/Model.pth'.format(p)
model = torch.load(model)

data ="FLIR"
print(count_parameters_in_MB(model))
image_VIS_list = prepare_data1(r'D:\Image_Data\IRVI\AUIF Datasets\16x\Test_{}/'.format(data))
image_IR_list = prepare_data(r'D:\Image_Data\IRVI\AUIF Datasets\16x\Test_{}/'.format(data))
save_path = r'./Fused/'.format(p)
if os.path.exists(save_path):
    pass
else:
    os.makedirs(save_path)
source_name,save_name = name_list(r'D:\Image_Data\IRVI\MSRS-main\MSRS-main\train\ir/')
all_time = []

for i in range(len(image_VIS_list)):
    S = time.time()
    IR = load_image(image_IR_list[i])
    VIS= load_rgb(image_VIS_list [i])

    if not opt.cuda:
        model = model.to(device)

        IR = (IR).to(device) / 255
        VIS = (VIS).to(device) / 255
        VIS_y = RGB2Y(VIS)

    with torch.no_grad():
        Fused,fea_M,fea_map= model(VIS_y,IR)
        Fused_RGB = YCbCr2RGB(VIS, Fused)

    out = change(Fused_RGB )
    cv2.imwrite(os.path.join(save_path,str(i+1)+'.jpg'),out)
    
    print('mask'+str(i+1)+' has saved')
    all_time.append(time.time() - S)


print('Mean [%f], var [%f]'% (np.mean(all_time), np.std(all_time)))

