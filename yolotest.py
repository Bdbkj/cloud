import base64
import cv2
import numpy as np
import boto3
import os

# 配置文件路径
yolo_path = "/tmp/yolo_tiny_configs/"
labels_path = "coco.names"
weights_path = "yolov3-tiny.weights"
config_path = "yolov3-tiny.cfg"
confthres = 0.3
nmsthres = 0.1

s3_client = boto3.client('s3')
bucket_name = 'zeinawsbucket'
dynamodb = boto3.resource('dynamodb')
table_name = 'todos'

def download_yolo_files():
    if not os.path.exists(yolo_path):
        os.makedirs(yolo_path)
    s3_client.download_file(bucket_name, 'yolo_tiny_configs/coco.names', os.path.join(yolo_path, labels_path))
    s3_client.download_file(bucket_name, 'yolo_tiny_configs/yolov3-tiny.weights', os.path.join(yolo_path, weights_path))
    s3_client.download_file(bucket_name, 'yolo_tiny_configs/yolov3-tiny.cfg', os.path.join(yolo_path, config_path))

def get_labels(labels_path):
    lpath = os.path.join(yolo_path, labels_path)
    LABELS = open(lpath).read().strip().split("\n")
    return LABELS

def get_weights(weights_path):
    weightsPath = os.path.join(yolo_path, weights_path)
    return weightsPath

def get_config(config_path):
    configPath = os.path.join(yolo_path, config_path)
    return configPath

def load_model(configpath, weightspath):
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net

def do_prediction(image, net, LABELS):
    (H, W) = image.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    results = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confthres:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                results.append({
                    "label": LABELS[classID],
                    "confidence": float(confidence),
                    "x": x,
                    "y": y,
                    "width": int(width),
                    "height": int(height)
                })
    return results

def save_to_dynamodb(id, detected_objects):
    table = dynamodb.Table(table_name)
    response = table.put_item(
        Item={
            'id': id,
            'detected_objects': detected_objects
        }
    )
    return response

def lambda_handler(event, context):
    try:
        # 下载YOLO文件
        download_yolo_files()

        # 获取输入数据
        image_b64 = event['image']
        image_data = base64.b64decode(image_b64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 加载模型
        LABELS = get_labels(labels_path)
        CFG = get_config(config_path)
        Weights = get_weights(weights_path)
        net = load_model(CFG, Weights)

        # 进行预测
        results = do_prediction(image, net, LABELS)

        detected_objects = []
        for obj in results:
            detected_objects.append({
                "label": obj["label"],
                "accuracy": obj["confidence"],
                "rectangle": {
                    "left": obj["x"],
                    "top": obj["y"],
                    "width": obj["width"],
                    "height": obj["height"]
                }
            })

        # 保存到DynamoDB
        save_to_dynamodb(event['id'], detected_objects)

        return {
            'statusCode': 200,
            'body': {
                'id': event['id'],
                'objects': detected_objects
            }
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': {
                'error': "An error occurred while processing the request.",
                'message': str(e)
            }
        }
