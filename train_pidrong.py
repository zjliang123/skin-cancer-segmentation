import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from model.pidrong2 import Unet2d

# 定义UNet模型结构（根据之前的UNet定义，不变）

# 数据集类
class SkinDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        # 读取并转换图片为灰度图
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256))  # 调整为64x64像素

        # 读取掩码文件并调整大小
        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, (256, 256))  # 调整为64x64像素
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)  # 转换为RGB

        # 将掩码转换为分类
        mask = self.mask_to_class(mask)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # 增加一个维度，适配灰度图格式 (1, H, W)
        image = np.expand_dims(image, axis=0)

        return image, mask

    def mask_to_class(self, mask):
        # 创建一个新的掩码，初始化为0
        new_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        #
        # # HYP(254,246,242) -> class 1
        # new_mask[np.all(mask == [254, 246, 242], axis=-1)] = 1
        #
        # # INF, FOL, RET, PAP, EPI, KER -> class 2
        # class2_colors = [[145, 1, 122], [216, 47, 148], [181, 9, 130], [236, 85, 157], [73, 0, 106], [248, 123, 168]]
        # for color in class2_colors:
        #     new_mask[np.all(mask == color, axis=-1)] = 2
        #
        # # BCC, SCC, IEC -> class 3
        # class3_colors = [[127, 255, 255], [127, 255, 142], [255, 127, 127]]
        # for color in class3_colors:
        #     new_mask[np.all(mask == color, axis=-1)] = 3

        # Background (0, 0, 0) -> class 0
        new_mask[np.all(mask == [108,0,115], axis=-1)] = 0
        new_mask[np.all(mask == [145,1,122], axis=-1)] = 1
        new_mask[np.all(mask == [216,47,148], axis=-1)] = 2
        new_mask[np.all(mask == [181,9,130], axis=-1)] = 3
        new_mask[np.all(mask == [236,85,157], axis=-1)] = 4
        new_mask[np.all(mask == [73,0,106], axis=-1)] = 5
        new_mask[np.all(mask == [248,123,168], axis=-1)] = 6
        new_mask[np.all(mask == [0, 0, 0], axis=-1)] = 7
        new_mask[np.all(mask == [254, 246, 242], axis=-1)] =8
        class3_colors = [[127, 255, 255], [127, 255, 142], [255, 127, 127]]
        for color in class3_colors:
            new_mask[np.all(mask == color, axis=-1)] =9
        return new_mask


# 创建数据集
train_dataset = SkinDataset(
    image_dir='./data/Training_Images',
    mask_dir='./data/Training_Masks',
    transform=None  # 你可以根据需要添加变换
)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 初始化模型
model = Unet2d(1, 10).cuda()  # 输入通道改为1（灰度图）

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  # 增加weight decay防止过拟合

# Tensorboard记录
writer = SummaryWriter('runs/skin_segmentation')
torch.autograd.set_detect_anomaly(True)
# 训练模型
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")

    for images, masks in progress_bar:
        # images = torch.tensor(images, dtype=torch.float32).cuda()  # 转换成浮点类型
        # masks = torch.tensor(masks, dtype=torch.long).cuda()
        images = images.float().cuda()  # 转换为浮点类型
        masks = masks.long().cuda()  # 转换为长整型

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, masks)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix({"Loss": running_loss / len(progress_bar)})

    # 每个epoch记录
    writer.add_scalar('Training Loss', running_loss / len(train_loader), epoch)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# 保存模型
torch.save(model.state_dict(), 'result_pidrong/unet_skin_segmentation.pth')

writer.close()
