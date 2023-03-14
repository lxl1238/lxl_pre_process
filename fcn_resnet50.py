import torch
import cv2
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.models.segmentation import fcn_resnet50

# 加载预训练的FCN-ResNet50模型
model = fcn_resnet50(pretrained=True).eval()

# 加载测试图像
img = cv2.imread('test-1.jpg')

# 对图像进行预处理
# 双边滤波
bilateral = cv2.bilateralFilter(img, 17, 100, 100)

# 显示滤波效果
cv2.imshow('Gaussian',bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 将分割结果转换成numpy array并显示
img = TF.to_tensor(bilateral).unsqueeze(0)

# 使用模型进行分割
with torch.no_grad():
    output = model(img)['out']
    mask = (output.argmax(1) == 15).float().unsqueeze(1)  # 提取头部区域

# 将分割结果转换成numpy array并显示
mask = mask.numpy()[0][0]

# 将灰度图转换为 BGR 颜色空间
mask = cv2.cvtColor(np.uint8(mask*255), cv2.COLOR_GRAY2BGR)
cv2.imshow('result', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 找到头部区域的最高点坐标
contours, hierarchy = cv2.findContours(np.uint8(mask[:,:,0]), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(contours, key=cv2.contourArea)
cnt = cnt.squeeze()

# 找到所有y轴坐标最小的点
min_y = np.min(cnt[:, 1])
min_points = []
for point in cnt:
    if point[1] == min_y:
        min_points.append(point)

img_draw = img[0].numpy().transpose(1, 2, 0).copy()
for point in min_points:
    cv2.circle(img_draw, tuple(point), 5, (0, 0, 255), -1)

cv2.imshow('result', img_draw)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(min_y)



