import time

import cv2
import ultralytics
import numpy as np
import streamlit as st
from skimage import exposure
import supervision as sv


hands_model = ultralytics.YOLO("Demo7_hands_segV3.pt")
#hands_model = ultralytics.YOLO("Demo7_hands_detectionsV1.pt")
#hands_model = ultralytics.YOLO("yolov8s.pt")
tracker = sv.ByteTrack(track_thresh=0.65,track_buffer=60,match_thresh=0.8, frame_rate=20)
gain = 1.4 #factor to scale up polygons and circles
COLORS = sv.ColorPalette.default()
bounding_box_annotator = sv.BoundingBoxAnnotator(color=COLORS)
mask_annotator = sv.MaskAnnotator(opacity=0.5)
trace_annotator = sv.TraceAnnotator(color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2)
label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_RIGHT,text_padding=5)
if 'idx' not in st.session_state:
        st.session_state['idx'] = 0

if 'image' not in st.session_state:
        st.session_state['image'] = "YorkATS logo.png"


def Capture_image():
        cv2.imwrite('/Users/tobieabel/Desktop/Demo7/magic_wands_spell_checker/' + str(idx) + '.jpeg',st.session_state.image)

with st.sidebar:
    st.button(':green[Capture Image]', key='images', on_click=Capture_image)

def calculate_coverage():
        dist_compare = {}
        prime_detection = 0
        Acc_Total = 0
        #for every detection currently in the custom tracker, calculate the cumulative change in xy coordinates to determine which detection has moved the most - this is an indicator of a spell being cast
        for i, v in custom_tracker.items():
                if len(v[2]) > 9: #v[2] is the list of centre points of each detection
                        sublist= v[2][-10:]
                        Acc_x = 0
                        Acc_y = 0
                        Acc_Total = 0
                        x_list = []
                        y_list = []
                        for centre in sublist:
                                x_list.append(centre[0][0])
                                y_list.append(centre[0][1])
                        for x in range((len(x_list)-1)):
                                start_x = x_list[x]
                                #print(start_x)
                                end_x = x_list[x+1]
                                #print(end_x)
                                start_y = y_list[x]
                                #print(start_y)
                                end_y = y_list[x+1]
                                #print(end_y)

                                dist_x = start_x-end_x
                                if dist_x < 0:
                                        dist_x = dist_x * -1
                                Acc_x += dist_x
                                dist_y = start_y-end_y
                                if dist_y < 0:
                                        dist_y = dist_y * -1
                                Acc_y += dist_y
                                Acc_Total +=(dist_x + dist_y)

                                #print(Acc_x)
                                #print(Acc_y)
                                #print(Acc_Total)
                                dist_compare[i] = Acc_Total
        #get the detection in dist_compare dictionary with the highesst Accumulated Total of X,Y movements
        if len(dist_compare) > 0:
                print(dist_compare)
                prime_detection = max(dist_compare)
                print(prime_detection)

        return prime_detection, round(Acc_Total,2)



custom_tracker = {} #custom tracker with tracker ID as name, and corresponding polygon zone, expiry count,list of centre points, polygon array and detection area as values
def custom_track_update(obj_detections):
        #check if tracker ID already in your custom tracker
        for i, tracker_id in enumerate(obj_detections.tracker_id):
                if tracker_id in custom_tracker:
                        #add the centre point to your custom tracker, and add 2 to expire count up to max 10
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
                                        #print(trigger)

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

def update_custom_tracker_zone(i,centre):
        centre_x = int(centre[0][0])
        centre_y = int(centre[0][1])
        x0 = int(centre_x * gain)
        y0 = int(centre_y * gain)
        x1 = int(centre_x - (x0 - centre_x))
        y1 = int(centre_y - (y0 - centre_y))
        polygon_array = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        polygon_zone = sv.PolygonZone(polygon_array, frame_resolution_wh=[640, 480],
                                      triggering_position=sv.Position.CENTER)
        custom_tracker[i][0] = polygon_zone
        custom_tracker[i][3] = polygon_array

sliding_window = []
for i in range(20):
        sliding_window.append(0)
def sliding_window_rule(obj_detections): #check how often the current highest confidence detection has been in the last 20 frames
        #print(obj_detections.tracker_id)
        if len(obj_detections.tracker_id) > 0:
                sliding_window.append(obj_detections.tracker_id[0]) #add the tracker ID to the list
                sliding_window.pop(0) #remove index 0 from the list
                #print(sliding_window)
                acc = 0
                for i in sliding_window: #go through the list and count how many time current detection occurs
                        if i == obj_detections.tracker_id[0]:
                                acc += 1
                if acc > 9: #if its 10 or more then we plot the detection, else false
                        return True
                else:
                        return False
        else:
                return False


def filter_highest_conf(obj_detections) -> sv.detection:
        if obj_detections.__len__(): #if there are any detections, find the highest confidence one
                highest_confidence_score = 0
                index = 0
                for i, confidence in enumerate(obj_detections.confidence):
                        if confidence > highest_confidence_score:
                                highest_confidence_score = confidence
                                index = i
                obj_detections = obj_detections[index] #filter the detections so only the highest confidence one remains
        return obj_detections

def find_centre_point() -> list:
        detection_circles = []

        for i, detection in enumerate(obj_detections):
                area = obj_detections.area[i]
                centre = obj_detections[i].get_anchors_coordinates(anchor=sv.Position("CENTER"))
                centre_x = int(centre[0][0])
                centre_y = int(centre[0][1])
                radius = int(np.sqrt(area*(gain))) #get radius of circle based on square root of detection (i.e. relative to size of detection which is relative to distance from camera
                #create a polygon zone based on area
                x0 = int(centre_x * gain)
                y0 = int(centre_y * gain)
                x1 = int(centre_x - (x0-centre_x))
                y1 = int(centre_y - (y0-centre_y))
                polygon = [[x0, y0],[x1, y0], [x1, y1], [x0, y1]]
                detection_circles.append([centre_x,centre_y, radius, polygon])
        return detection_circles


LiveStream = st.image("YorkATS logo.png",channels="RGB", width=640)
LiveStream2 = st.image("YorkATS logo.png",channels="RGB", width=640)
#put each frame into a queue to create a buffer
#queue code goes here
#while True:
#    obj_det_result = hands_model.predict(source='https://www.youtube.com/watch?v=5EWot--bBOw&ab_channel=TOBIEABEL', stream=True)
#if you get an error regarding certificates it will be something to do with Python - go to python folder > double click on "Install Certificates.command" file
#    for r in obj_det_result:
#        im_array = r.plot()  # use Ultralytics library to plot a BGR numpy array of predictions
#        youtube_frame = cv2.resize(im_array,(640,480))
#        youtube_frame = cv2.cvtColor(youtube_frame, cv2.COLOR_BGR2RGB)
#        LiveStream.image(youtube_frame, channels="RGB", width=640)

cap = cv2.VideoCapture("/Users/tobieabel/PycharmProjects/Demo-v7/magic wands.mp4")
idx = 0
while cap.isOpened():
    ret, frame1 = cap.read()
    idx += 1
    if idx % 1 == 0:
#resize
        frame1 = cv2.resize(frame1, (640, 480))

#preprocess frame ---------------
#grey and blur images
#gray_image = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
#blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
#blurred_image = np.uint8(blurred_image)

#get shaper edges with Laplacian - need to try this with colour image
#frame2 = cv2.Laplacian(blurred_image, cv2.CV_64F)
#frame2 = np.uint8(np.abs(frame2))
#sharpened_image = cv2.addWeighted(blurred_image, 1.0, frame2, -0.5, 0)

#use histogram equalisation to enhance the contrast
#equalized_image = cv2.equalizeHist(sharpened_image)

#use histogram equalisation on colour images - doesn't need above grey and blur steps
        yuv_image = cv2.cvtColor(frame1, cv2.COLOR_BGR2YUV)
        yuv_image[:,:,0] = cv2.equalizeHist(yuv_image[:,:,0])
        equalized_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

#local histogram equalisation - enhance contrast in local regions - couldn't get this to work
#equalized_image = exposure.equalize_hist(blurred_image)
#equalized_image = np.uint8(np.abs(equalized_image))


#Unsharp Masking
#unsharp_mask = cv2.addWeighted(gray_image, 1.5, blurred_image, -0.5, 0)

#bilateral filter
#filtered_image = cv2.bilateralFilter(frame1, d=9, sigmaColor=75, sigmaSpace=75)

#errorsion
#kernel = np.ones((5, 5), np.uint8)
#erosion = cv2.erode(gray_image, kernel, iterations=1)

#dilation
#dilation = cv2.dilate(gray_image, kernel, iterations=1)

# Set a threshold value (adjust as needed)
#threshold_value = 5
#mask = gray_image < threshold_value #less than the threshold turns dark areas white
#gray_image[mask] = [255]
#mask2 = gray_image > 25 #greater than the threshold turns pale areas white
#gray_image[mask2] = [255]
#unsharp_mask = cv2.addWeighted(gray_image, 1.5, blurred_image, -0.5, 0)

#------------------------------------------
        #save image for training
        #cv2.imwrite('/Users/tobieabel/Desktop/Demo7/magic_wands_vid_v2/' + str(idx) + 'v2.jpeg',equalized_image)


        #run object detections for staff
        obj_det_result = hands_model.predict(source=equalized_image, device="mps", verbose=False, conf=0.65, iou=0.65)[0]
        obj_detections = sv.Detections.from_ultralytics(obj_det_result)

        obj_detections = tracker.update_with_detections(obj_detections)
        #fix tracker with custom polygon zone tracking because Byte_track keeps swapping ID's even for static detections
        #for each tracker id, check if its in your custom_traker and if not has it triggered and existing trackers zone
        track_id, custom_id = custom_track_update(obj_detections)
        #if track_id != 0:    #how do you edit a detections object?

        #replace confidence filter with movement fiter once tracker is fixed
        #obj_detections = filter_highest_conf(obj_detections)#filter results to get highest confidence detection only


        #if obj_detections.__len__:
        #       sliding_window_flag = sliding_window_rule(obj_detections)  # only detections that have been around for at least x frames are kept

        #annotate image
        #use your own custom tracker to annotate frames with custom tracker ID(from current frame only) and trace of all those centre points
        #need to just plot the current centre point with circle, not bounding box

        prime_detection, Acc_Total = calculate_coverage() #calculate the detection that has moved the most in the last 10 frames

        for i,v in custom_tracker.items():
                if i == prime_detection and Acc_Total > 55: #only show the annotations for the hand that moved the most, and only if it moved over 55 pixels in total
                        centre_point = v[2][-1] #get the latest centre point coordinates
                        area = v[4]
                        centre_x = int(centre_point[0][0])
                        centre_y = int(centre_point[0][1])
                        radius = int(np.sqrt(area * gain))
                        #cv2.circle(equalized_image, (centre_x, centre_y), radius, (0, 0, 255), 2)
                        update_custom_tracker_zone(i,centre_point)
                        #sv.draw_polygon(equalized_image, v[3], COLORS.colors[0])
                        #sv.draw_text(equalized_image,str(i),sv.Point(centre_x,centre_y),COLORS.colors[1],text_scale=1, text_thickness=1)
                        #sv.draw_text(equalized_image, "Prime Detection = " + str(Acc_Total) , sv.Point(centre_x, (centre_y - 50)), COLORS.colors[0],text_scale=0.75, text_thickness=2)
                        #draw tracer from custom tracker - need to convert the points to int32 and put them into a list (even though they are a list already!
                        cv2.polylines(equalized_image, np.int32([v[2]]), isClosed=True, color=(0, 0, 255), thickness=2)



        #print("----------------------------------------------------------------------------")

        equalized_image = bounding_box_annotator.annotate(equalized_image, obj_detections)
        #equalized_image = mask_annotator.annotate(equalized_image, obj_detections)
        #equalized_image = trace_annotator.annotate(equalized_image,obj_detections)
        #for i, id in enumerate(obj_detections.tracker_id):
        #        equalized_image = label_annotator.annotate(equalized_image,detections=obj_detections[i],labels=[f'#{id} {obj_det_result.names[obj_detections.class_id[i]]}'])

        #show image on streamlit
        final_frame = cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB)
        LiveStream.image(final_frame, channels="RGB", width=640)
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        LiveStream2.image(frame1, channels="RGB", width=640)
        #time.sleep(0.2)
        st.session_state.image = equalized_image
        st.session_state.idx += 1

#streamlit run /Users/tobieabel/PycharmProjects/Demo-v7/magic_wands.py

