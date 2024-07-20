from ultralytics import YOLO
if __name__  == "__main__":
    # -------------------------数据配置-----------------------------------------
    data="/home/neuedu/桌面/yolov8/config/dataset/undown.yaml"
    project="/home/neuedu/桌面/yolov8/runs/yolo" # 项目保存路径
    # -------------------------改进模型YOLOv8n-----------------------------------------
    yolov8n = "/home/neuedu/桌面/yolov8/runs/yolo/yolov8n_baseline/weights/last.pt"
    yolov8n_name = "yolov8n_baseline"
    for batchs in [8,16,32]:
        for optimizer in ['SGD', 'Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'auto']:
            yolov8n = f"/home/neuedu/桌面/yolov8/runs/yolos/yolov8n_our{str(batchs)}{optimizer}/weights/best.pt"
            model = YOLO(yolov8n)
            print(f"batchs:{batchs}  optimizer:{optimizer} \n")
            results = model.val(data=data ,imgsz=640, batch=1 ,workers = 8)
            print(results)


