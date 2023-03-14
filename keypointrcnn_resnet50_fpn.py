import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# 判断是否可以使用 GPU，如果可以则使用 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练模型
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device)

# 读取图像并进行预处理
image = Image.open("test-1.jpg")
transform = transforms.Compose([transforms.ToTensor()])
image_t = transform(image).to(device)

# 将模型设置为评估模式，即不进行训练
model.eval()

# 使用模型进行预测
with torch.no_grad():
    pred = model([image_t])

#取得头像边框的顶部纵坐标
for i in range(len(pred[0]["labels"])):
    if pred[0]["labels"][i] != 1:
        continue
    else:
        value = pred[0]["boxes"][i][1].item()
        break

print(pred)
print(value)

# 定义 COCO 数据集中的类别名称
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 将预测结果转换成可视化的结果
pred_class = [COCO_INSTANCE_CATEGORY_NAMES[ii] for ii in list(pred[0]['labels'].cpu().numpy())]
pred_score = list(pred[0]['scores'].detach().cpu().numpy())
pred_boxes = [[ii[0], ii[1], ii[2], ii[3]] for ii in list(pred[0]['boxes'].detach().cpu().numpy())]
pred_index = [pred_score.index(x) for x in pred_score if x > 0.5]
fontsize = np.int16(image.size[1] / 30)

# 使用 PIL 库绘制结果
draw = ImageDraw.Draw(image)
for i, index in enumerate(pred[0]['labels']):
    if pred[0]['scores'][i] > 0.5:
        box = pred[0]['boxes'][i]
        label = COCO_INSTANCE_CATEGORY_NAMES[index]
        score = pred[0]['scores'][i]
        draw.rectangle(box.tolist(), outline="red")
        draw.text((box[0], box[1]), f"{label}: {score:.2f}", fill="red")

# show the image
plt.figure("dog")
plt.axis('off')  # 关掉坐标轴为 off
plt.imshow(image)
plt.show()
