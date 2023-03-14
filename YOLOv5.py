import torch
import torchvision
import PIL.Image as Image
import matplotlib.pyplot as plt

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

image = Image.open('test-1.jpg')
results = model(image)
print(model)
print(model)
# 获取所有检测到的目标的类别标签、置信度、位置和面积等信息
labels = results.xyxy[0][:, -1].cpu().numpy()
scores = results.xyxy[0][:, -2].cpu().numpy()
boxes = results.xyxy[0][:, :-2].cpu().numpy()

areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

# 绘制检测结果
fig, ax = plt.subplots(1)
ax.imshow(image)
for label, score, box in zip(labels, scores, boxes):
    ax.add_patch(plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none'))
    ax.text(box[0], box[1], f'{label} {score:.2f}', fontsize=6, color='white')
plt.show()

