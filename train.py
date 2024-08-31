from ultralytics import YOLO
if __name__  == "__main__":
    data="configuration file"
    model = YOLO(‘configuration file’)
    # print(model)
    for batchs in [8,16,32]:
        for optimizer in ['SGD', 'Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'auto']:
            model.train(data=data,epochs=300 ,imgsz=640,optimizer=optimizer, batch=batchs ,workers = 8,project=project,name=yolov8n_name + str(batchs) +'-' + optimizer)