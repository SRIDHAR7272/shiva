import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

classNames = []
classFile = "C:/Users/Lenovo/Desktop/obj_drone/Object_Detection_Files/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

model_path = "Object_Detection_Files/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def getObjects(img, thres, draw=True, objects=[]):
    # Prepare the image
    height, width, _ = img.shape
    input_shape = input_details[0]['shape']
    input_data = cv2.resize(img, (input_shape[2], input_shape[1]))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = (np.float32(input_data) - 127.5) / 127.5

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get output tensor
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    class_ids = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    objectInfo = []
    for i in range(len(scores)):
        if scores[i] > thres:
            class_id = int(class_ids[i])
            class_name = classNames[class_id]
            if len(objects) == 0 or class_name in objects:
                box = boxes[i]
                ymin, xmin, ymax, xmax = box
                xmin, xmax, ymin, ymax = int(xmin * width), int(xmax * width), int(ymin * height), int(ymax * height)
                bbox = xmin, ymin, xmax - xmin, ymax - ymin
                objectInfo.append([bbox, class_name])
                if draw:
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=2)
                    cv2.putText(img, class_name.upper(), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(scores[i] * 100, 2)), (xmin, ymax + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return img, objectInfo


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        success, img = cap.read()
        result, objectInfo = getObjects(img, 0.45, objects=['person'])
        cv2.imshow("Output", img)
        cv2.waitKey(1)
