import cv2
import ultralytics
import numpy as np
import streamlit as st
import supervision as sv
import time



LiveStream5 = st.image("YorkATS logo.png",channels="RGB", width=640)
COLORS = sv.ColorPalette.default()
hands_model = ultralytics.YOLO("Demo7_SeanceV2.pt")
Class_model = ultralytics.YOLO("Demo7_class_v3.pt")
bounding_box_annotator = sv.BoundingBoxAnnotator(color=COLORS)
label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_RIGHT)
tracker = sv.ByteTrack(track_thresh=0.65,track_buffer=30,match_thresh=0.8, frame_rate=20)
trace_annotator = sv.TraceAnnotator(color=COLORS, position=sv.Position.CENTER, trace_length=60, thickness=2)

Timer = 0
custom_tracker = {}
Broken_Circle = False
prime_detection = None
def update_tracker(hands_detection):
    global custom_tracker, Timer

    Timer -= 1
    #loop through the latest hands detection and record the centre point for each
    for i, detection in enumerate(hands_detection):
        centre = hands_detection[i].get_anchors_coordinates(anchor=sv.Position("CENTER"))
        if detection[4] in custom_tracker: #if the tracker ID is already in the custom tracker, add the centre coordinates
            custom_tracker[detection[4]].append(centre[0])
        else: # otherwise add this tracker ID to the custom tracker
            custom_tracker[detection[4]] = [centre[0]]


def calculate_coverage():
    global Timer, prime_detection
    Timer -= 1
    dist_compare = {}

    x_list = []
    y_list = []
    Acc_x = 0
    Acc_y = 0
    Acc_Total = 0
    for i, v in custom_tracker.items():
        x_list = []
        y_list = []
        for centre in v:
            x_list.append(centre[0]) #list of all x coordinates for this tracker ID over last 10 frames
            y_list.append(centre[1])

        for x in range((len(x_list) - 1)):
            start_x = x_list[x]
            # print(start_x)
            end_x = x_list[x + 1]
            # print(end_x)
            start_y = y_list[x]
            # print(start_y)
            end_y = y_list[x + 1]
            # print(end_y)

            dist_x = start_x - end_x
            if dist_x < 0:
                dist_x = dist_x * -1
            Acc_x += dist_x
            dist_y = start_y - end_y
            if dist_y < 0:
                dist_y = dist_y * -1
            Acc_y += dist_y
            Acc_Total += (dist_x + dist_y)

            dist_compare[i] = Acc_Total
            Acc_x = 0
            Acc_y = 0
            Acc_Total = 0


    if len(dist_compare) > 0:
        print(dist_compare)
        prime_detection = max(dist_compare, key=dist_compare.get)
        print(prime_detection)

    return (prime_detection)



cap = cv2.VideoCapture("/Users/tobieabel/PycharmProjects/Demo-v7/seance 2a.mp4")
idx = 0
while cap.isOpened():
    ret, frame1 = cap.read()
    idx += 1

    #preprocess frame
    frame1 = cv2.resize(frame1, (640, 480))


    #run object detection model
    obj_det_result = hands_model.predict(source=frame1, device="mps", verbose=False, conf=0.65, iou=0.65)[0]
    obj_detections = sv.Detections.from_ultralytics(obj_det_result)
    if len(obj_detections) > 0:
        obj_detections = tracker.update_with_detections(obj_detections)
        hand_detections = obj_detections[obj_detections.class_id==1] #filter the detections to just give the hands
        frame1 = bounding_box_annotator.annotate(frame1, hand_detections)
        labels = [f"#{id} {obj_det_result.names[class_id]}" for id, class_id in
                  zip(hand_detections.tracker_id, hand_detections.class_id)]  # remember that labels have to be a list
        frame1 = label_annotator.annotate(scene=frame1, detections=hand_detections, labels=labels)
        frame1 = trace_annotator.annotate(scene=frame1, detections=hand_detections)


    # run classification model to see if circle is broken - trained model in colab but it didn't work!, tried the following
    # not resize issue
    # not PIL image type issue
    # old model works
    # removed redundant 'unlabelled class'
    # ensured train and validate had examples of each class
    # trained old dataset from demo 5 for new model
    # upgraded Ultralytics to same version as in colab - this fixed it!
    if Timer == 0 and Broken_Circle == False: #if timer is 0 this means we are not in the middle of a broken circle loop or displaying broken circle

        class_results = Class_model(frame1, device='mps')[0]
        if class_results.probs.top1 == 0: #'0' = broken, 1 = circle, 2 = none
            Timer = 10
            custom_tracker = {}
            update_tracker(hand_detections)

    elif Broken_Circle:
        class_results = Class_model(frame1, device='mps')[0]
        if class_results.probs.top1 != 0: #The circle was broken but is now either a circle or none
            Broken_Circle = False


    elif Timer == 1: #this is the final loop for broken circles so calculate which hand has moved most
        prime_detection = calculate_coverage()
        if prime_detection != None:
            Broken_Circle = True



    else: # we're in the middle of a broken circle loop so update custom tracker
        print(Timer)
        update_tracker(hand_detections)



    if Broken_Circle:
        centre_point = custom_tracker[prime_detection][-1]
        print(centre_point)
        text_anchor = sv.Point(x=int(centre_point[0]), y=int(centre_point[1]))
        frame1 = sv.draw_text(scene=frame1, text="The Circle Has Been Broken!", text_anchor=text_anchor,
                              text_thickness=1, text_padding=5,
                              background_color=COLORS.colors[0])

    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    LiveStream5.image(frame1, channels="RGB", width=640)




#streamlit run /Users/tobieabel/PycharmProjects/Demo-v7/magic_wands.py
