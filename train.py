from ultralytics import RTDETR
if __name__  == "__main__":
    data="/home/neuedu/桌面/yolov8/config/dataset/undown.yaml"
    project="/home/neuedu/桌面/yolov8/runs/rt-detr"
    yolov8n = "/home/neuedu/桌面/yolov8/ultralytics/cfg/models/rt-detr/xxx.yaml"
    yolov8n_name = "rt-detr"
    model = RTDETR(yolov8n)
    # print(model)
    for batchs in [8,16,32]:
        for optimizer in ['SGD', 'Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'auto']:
            model.train(data=data,epochs=300 ,imgsz=640,optimizer=optimizer, batch=batchs ,workers = 8,project=project,name=yolov8n_name + str(batchs) +'-' + optimizer)