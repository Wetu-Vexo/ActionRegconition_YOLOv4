import numpy as np
import cv2

def extract_boxes_confidences_classids(outputs, confidence, width, height):
    boxes = []
    confidences = []
    classIDs = []

    for output in outputs:
        for detection in output:            
            # Extract the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classID = np.argmax(scores)
            conf = scores[classID]
            
            # Consider only the predictions that are above the confidence threshold
            if conf > confidence:
                # Scale the bounding box back to the size of the image
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, w, h = box.astype('int')

                # Use the center coordinates, width and height to get the coordinates of the top left corner
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(conf))
                classIDs.append(classID)

    return boxes, confidences, classIDs


def draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors):
    if len(idxs) > 0:
        for i in idxs.flatten():
            # extract bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            # draw the bounding box and label on the image
            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

def make_prediction(net, layer_names, labels, image, confidence, threshold):
    height, width = image.shape[:2]
    # Pre-processar a imagem para ela tornar-se blob
    # Passar pelo modelo Yolo
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (320, 320), swapRB=True, crop=False)
    #blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    #blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (512, 512), swapRB=True, crop=False)
    #blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (608, 608), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)

    # Extrair os retangulos, confianca e classIDs
    boxes, confidences, classIDs = extract_boxes_confidences_classids(outputs, confidence, width, height)

    # Aplicar Non-Max Suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

    return boxes, confidences, classIDs, idxs

def is_touching_line(line_start, line_end, object_center):
    # Check if the object's x-coordinate is within the range of the line's x-coordinates
    if line_start[0] <= object_center[0] <= line_end[0]:
        # Calculate the corresponding y-coordinate on the line for the object's x-coordinate
        expected_y = line_start[1] + ((object_center[0] - line_start[0]) / (line_end[0] - line_start[0])) * (line_end[1] - line_start[1])
        
        # Check if the object's y-coordinate is close enough to the calculated y-coordinate on the line
        if abs(object_center[1] - expected_y) <= 2:  # You can adjust the tolerance value as needed
            return True
        
    return False
# Objetos que o modelo detecta
labels = open('coco.names').read().strip().split('\n')

# Gerar cores aleatoriamente para cada categoria de objeto
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
#video path
video_path = 0
# Carregar o modelo e os pesos
net = cv2.dnn.readNetFromDarknet('custom-yolov4-tiny-detector.cfg', 'custom-yolov4-tiny-detector_best.weights')

use_gpu = 1
if (use_gpu == 1):
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Obter o nome das categorias
layer_names = net.getLayerNames()
layer_names = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]


cap = cv2.VideoCapture(video_path)
# define variable for record video

output_video_path = "output_video.avi"

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#fps = int(cap.get(cv2.CAP_PROP_FPS))
fps = 5
codec = cv2.VideoWriter_fourcc(*'MP4V')
output_video_path = "output_video_yolov4.mp4"
out = cv2.VideoWriter(output_video_path, codec, fps, (width, height))
#cap = cv2.VideoCapture('0')
stop = 0

# Initialize a list to keep track of displayed classes
displayed_classes = []

while True:
    if stop == 0:
        ret, frame = cap.read()
        if ret:
            boxes, confidences, classIDs, idxs = make_prediction(net, layer_names, labels, frame, 0.1, 0.3)
            frame = draw_bounding_boxes(frame, boxes, confidences, classIDs, idxs, colors)
            
            # Iterate through detected objects and add class names to the list
            for i in range(len(boxes)):
                class_id = classIDs[i]
                confidence = confidences[i]
                class_name = labels[class_id]

                # Add the class name to the list of displayed classes if not already present
                if class_name not in displayed_classes:
                    displayed_classes.append(class_name)

            # Display the detected class names at the top of the frame
            text = ", ".join(displayed_classes)
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow("Frame", frame)
            out.write(frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                stop = not stop
        if key == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()


