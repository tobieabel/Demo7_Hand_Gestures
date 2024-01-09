import cv2
import ultralytics
import numpy as np
import streamlit as st
import supervision as sv


LiveStream3 = st.image("YorkATS logo.png",channels="RGB", width=640)
LiveStream4 = st.image("YorkATS logo.png",channels="RGB", width=640)
COLORS = sv.ColorPalette.default()
hands_model = ultralytics.YOLO("Demo7_hands_segV3.pt")
bounding_box_annotator = sv.BoundingBoxAnnotator(color=COLORS)
label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_RIGHT)
tracker = sv.ByteTrack(track_thresh=0.65,track_buffer=60,match_thresh=0.8, frame_rate=20)
trace_annotator = sv.TraceAnnotator(color=COLORS, position=sv.Position.CENTER, trace_length=60, thickness=2)

gain = 1.4
custom_tracker = {} #custom tracker with tracker ID as name, and corresponding polygon zone, expiry count,list of centre points, polygon array and detection area as values
def custom_track_update(obj_detections):
        #check if tracker ID already in your custom tracker
        for i, tracker_id in enumerate(obj_detections.tracker_id):
                if tracker_id in custom_tracker:
                        #add the centre point to your custom tracker, and add 2 to expire count
                        centre = obj_detections[i].get_anchors_coordinates(anchor=sv.Position("CENTER"))
                        centre_x = int(centre[0][0])
                        centre_y = int(centre[0][1])
                        custom_tracker[tracker_id][2].append(centre)
                        custom_tracker[tracker_id][1] += 2

                else:
                        #check if existing tracker zone is triggered by the detection
                        if len(custom_tracker) > 0:
                                copy_custom_tracker = custom_tracker.copy() #have to loop through a copy as you changing dictionary size in the loop below
                                triggered_flag = False
                                for idx, values in copy_custom_tracker.items():
                                        trigger = values[0].trigger(obj_detections[i])

                                        if trigger == [True]: #detection is within existing zone so not new
                                                #check if the tracker ID of the triggered zone is in this frames detections
                                                #if it is then ignore this tracker ID as we want to follow official tracker ID
                                                triggered_flag = True
                                                if idx in obj_detections.tracker_id:
                                                        #print("tracker ID ", tracker_id, " triggered zone ", idx, " which is in this frame")
                                                        break #don't want to cycle through anymore zones for this detection

                                                else: #if it isn't then add the centre point of this ID to the official ID and 2 to expiry count as this detection is likley a swap.
                                                        #print("tracker ID ", tracker_id, " triggered zone ", idx," which is NOT in this frame")
                                                        centre_existing = obj_detections[i].get_anchors_coordinates(
                                                                anchor=sv.Position("CENTER"))
                                                        custom_tracker[idx][2].append(centre_existing)
                                                        custom_tracker[idx][1] += 2
                                                        break

                                if triggered_flag == False: #this is entirely new detection so add it to the custom tracker
                                        #calculate polygon
                                        centre = obj_detections[i].get_anchors_coordinates(anchor=sv.Position("CENTER"))
                                        centre_x = int(centre[0][0])
                                        centre_y = int(centre[0][1])
                                        x0 = int(centre_x * gain)
                                        y0 = int(centre_y * gain)
                                        x1 = int(centre_x - (x0 - centre_x))
                                        y1 = int(centre_y - (y0 - centre_y))
                                        polygon_array = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
                                        #initiate polygonzone
                                        polygon_zone = sv.PolygonZone(polygon_array, frame_resolution_wh=[640, 480],
                                                                 triggering_position=sv.Position.CENTER)
                                        area = obj_detections.area[i]
                                        custom_tracker[tracker_id] = [polygon_zone, 10, [centre], polygon_array, area]


                        else:  # if there is currently no detections in your custom tracker then just add it
                                # calculate polygon
                                centre = obj_detections[i].get_anchors_coordinates(anchor=sv.Position("CENTER"))
                                centre_x = int(centre[0][0])
                                centre_y = int(centre[0][1])
                                x0 = int(centre_x * gain)
                                y0 = int(centre_y * gain)
                                x1 = int(centre_x - (x0 - centre_x))
                                y1 = int(centre_y - (y0 - centre_y))
                                polygon_array = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
                                # initiate polygonzone
                                polygon_zone = sv.PolygonZone(polygon_array, frame_resolution_wh=[640, 480],
                                                              triggering_position=sv.Position.CENTER)
                                area = obj_detections.area[i]
                                custom_tracker[tracker_id] = [polygon_zone, 10, [centre], polygon_array, area]

        #reduce all expiry numbers by 1 and remove any that have expired (value 0 or less)
        custom_tracker1 = custom_tracker.copy()
        for i, v in custom_tracker1.items():
                v[1] -= 1
                if v[1] <= 0:
                        custom_tracker.pop(i)
                if len(v[2]) > 30: #remove centre points after 30 frames
                        v[2].pop(0)

        return (0,0) #might want to return trackerID and idx, but 0's for now


def calculate_coverage():
    dist_compare = {}
    prime_detection = 0
    Acc_Total = 0
    # for every detection currently in the custom tracker, calculate the cumulative change in xy coordinates to determine which detection has moved the most - this is an indicator of a spell being cast
    for i, v in custom_tracker.items():
        if len(v[2]) > 9: #if there are at least 10 centre positions recorded for this detection
            sublist = v[2][-10:] #get the last 10 centre positions
            Acc_x = 0
            Acc_y = 0
            Acc_Total = 0
            x_list = []
            y_list = []
            for centre in sublist:
                x_list.append(centre[0][0]) #record the x coordinates
                y_list.append(centre[0][1]) #and the y coordinates
            for x in range((len(x_list) - 1)):
                start_x = x_list[x]
                # print(start_x)
                end_x = x_list[x + 1]
                # print(end_x)
                start_y = y_list[x]
                # print(start_y)
                end_y = y_list[x + 1]
                # print(end_y)

                dist_x = start_x - end_x #calculate the distance between each x value in order
                if dist_x < 0:
                    dist_x = dist_x * -1 #flip it if its a negative number so we're dealing with all positive numbers
                Acc_x += dist_x
                dist_y = start_y - end_y
                if dist_y < 0:
                    dist_y = dist_y * -1 #same for y coordinates
                Acc_y += dist_y
                Acc_Total += (dist_x + dist_y) #and store the accumulated total of both x and y coordinate changes

                dist_compare[i] = Acc_Total #now store that total in a dictionary so you can compare all detection accumulated totals

    # get the detection in dist_compare dictionary with the highest Accumulated Total of X,Y movements
    if len(dist_compare) > 0:
        #print(dist_compare)
        prime_detection = max(dist_compare)
        #print(prime_detection)

    return prime_detection, round(Acc_Total, 2), dist_compare

def update_custom_tracker_zone(i, centre): #update both the polygon array coordinates and initiate a new polygon zone for each detection in the custom tracker
    centre_x = int(centre[0][0])
    centre_y = int(centre[0][1])
    x0 = int(centre_x + 140) #changing from using gain here to create more regular rectangle around detection
    y0 = int(centre_y + 140)
    x1 = int(centre_x - (x0 - centre_x))
    y1 = int(centre_y - (y0 - centre_y))
    polygon_array = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    polygon_zone = sv.PolygonZone(polygon_array, frame_resolution_wh=[640, 480],
                                  triggering_position=sv.Position.CENTER)
    custom_tracker[i][0] = polygon_zone
    custom_tracker[i][3] = polygon_array


cap = cv2.VideoCapture("/Users/tobieabel/PycharmProjects/Demo-v7/magic hands.mp4")
idx = 0
while cap.isOpened():
    ret, frame1 = cap.read()
    idx += 1

    #preprocess frame
    frame1 = cv2.resize(frame1, (640, 480))
    yuv_image = cv2.cvtColor(frame1, cv2.COLOR_BGR2YUV)
    yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])
    equalized_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)


    #run object detection model
    obj_det_result = hands_model.predict(source=equalized_image, device="mps", verbose=False, conf=0.45, iou=0.65)[0]
    obj_detections = sv.Detections.from_ultralytics(obj_det_result)
    obj_detections = tracker.update_with_detections(obj_detections)
    #equalized_image = bounding_box_annotator.annotate(equalized_image, obj_detections)
    #equalized_image = label_annotator.annotate(scene = equalized_image,detections = obj_detections, labels=[f'#{obj_detections.tracker_id}, {obj_detections.confidence}'])
    #equalized_image = trace_annotator.annotate(scene=equalized_image,detections=obj_detections)

    #update custom tracker
    custom_track_update(obj_detections)
    #print(custom_tracker)

    #Calculate quickest moving detection
    prime_detection, Acc_Total, dist_compare = calculate_coverage()
    #print(dist_compare)

    #update polygon zone of each detection in the custom tracker so zone moves as the hand moves
    for i, v in custom_tracker.items():
        #not using just the prime detection or filtering out other detections in this example
        centre_point = v[2][-1]  # get the latest centre point coordinates
        area = v[4]
        centre_x = int(centre_point[0][0])
        centre_y = int(centre_point[0][1])
        radius = int(np.sqrt(area * gain))
        #cv2.circle(equalized_image, (centre_x, centre_y), radius, (0, 0, 255), 2)
        update_custom_tracker_zone(i, centre_point)
        #sv.draw_polygon(equalized_image, v[3], COLORS.colors[0])
        #sv.draw_text(equalized_image,str(i),sv.Point(centre_x,centre_y),COLORS.colors[1],text_scale=1, text_thickness=1)
        # sv.draw_text(equalized_image, "Prime Detection = " + str(Acc_Total) , sv.Point(centre_x, (centre_y - 50)), COLORS.colors[0],text_scale=0.75, text_thickness=2)
        #draw tracer from custom tracker - need to convert the points to int32 and put them into a list (even though they are a list already!
        cv2.polylines(equalized_image, np.int32([v[2]]), isClosed=True, color=(0, 0, 255), thickness=2)


    final_frame = cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB)
    LiveStream3.image(final_frame, channels="RGB", width=640)
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    LiveStream4.image(frame1, channels="RGB", width=640)

#streamlit run /Users/tobieabel/PycharmProjects/Demo-v7/magic_wands.py
