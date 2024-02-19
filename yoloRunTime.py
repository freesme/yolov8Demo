from ultralytics import YOLO

model = YOLO('yolov8n.pt')
conf = 0.5
# 只检测人出现
classes = [0]
# 直播流视频
results = model.track(source="rtmp://mobliestream.c3tv.com:554/live/goodtv.sdp", stream=True, iou=0.9,
                      max_det=100, vid_stride=20, classes=classes)

for result in results:
    if result.boxes is not None:  # 检查是否存在边界框信息
        xyxy_boxes = result.boxes.xyxy  # 获取边界框的坐标信息
        conf_scores = result.boxes.conf  # 获取边界框的置信度信息
        cls_ids = result.boxes.cls  # 获取边界框的类别ID信息

        for box, conf, cls_id in zip(xyxy_boxes, conf_scores, cls_ids):
            x1, y1, x2, y2 = map(int, box)  # 将边界框的坐标转换为整数类型
            cls_id = int(cls_id)  # 将类别ID转换为整数类型
            label = model.names[cls_id]  # 根据类别ID获取对应的标签名称
            print("[x1:", x1, "y1", y1, "x2", x2, "y2", y2, "]-----------", label)
