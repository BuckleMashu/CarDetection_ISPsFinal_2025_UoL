# %%
!pip install opencv-contrib-python matplotlib numpy

# %%
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#these are the libraries used for Exercise_1
#.venv python 3.13.1

# %%
#task 1
cap = cv.VideoCapture('Traffic_Laramie_1.mp4')
if not cap.isOpened():
    print("Error opening video stream")
    exit()

min_contour_area = 1000

cars = {}
detected_location = []
car_counter = 0

background = None

while True:
    ret, frame = cap.read()
    if frame is None:
        break
        
    # Reduce frame size
    frame = cv.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
    
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray,(3,3),0)
    
    if background is None:
        background = blur
        
    difference = cv.absdiff(background,blur)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    dilation = cv.dilate(difference, kernel)
    fgmask = cv.threshold(dilation, 12, 255, cv.THRESH_BINARY)[1]

    # Define region of interest parameters
    height, width, _ = frame.shape
    bX = 0
    bY2 = height
    bY = int(height * 5/11)
    bX2 = width
    # draws a red line with thickness 1.
    # this indicate the area we care about
    cv.line(frame, (0, bY), (bX2, bY), (0,0,255), 1)
    
    # Find contours in the foreground mask
    contours, _ = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Process each contour
    for contour in contours:
        # if the contour is within the its area limit, it is most likely a car
        if min_contour_area < cv.contourArea(contour) < (height * width)*0.5:
            # Calculate centroid
            M = cv.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

            # Get bounding box coordinates
            x, y, w, h = cv.boundingRect(contour)
                
            # If the car's centroid is in the Main Street, draw a green rectangle around it
            if bX < cX < bX2 and bY < cY < bY2: # and min(h,w) >= 25
                cv.rectangle(frame, (x, y),(x + w, y + h), (0, 255, 0), 1)
    
    # Display the processed video output and the computer vision of the video
    cv.imshow('Bounding Box Frame', frame)
    cv.imshow('frame differenced', fgmask)
    
    # Check for exit command
    keyboard = cv.waitKey(30)
    if keyboard == ord('q') or keyboard == 27:
        break

cap.release()
cv.destroyAllWindows()

# %% [markdown]
# Task 1:
# *-----------Initialization and Setup:-----------*
# 
# cap = cv.VideoCapture('Traffic_Laramie_1.mp4'): 
# - Opens the video file for processing.
# 
# min_contour_area = 1000: 
# - Sets a minimum area threshold for contours. Contours smaller than this will be ignored (likely noise).
# 
# cars = {}, detected_location = [], car_counter = 0: 
# - Initializes variables to store car data (not actively used in this simplified version, but good practice).
# 
# background = None: 
# - Initializes a variable to store the background frame (an empty frame with no car).
# 
# *-----------Main Processing Loop (while True):-----------*
# 
# ret, frame = cap.read(): 
# - Reads the next frame from the video. ret indicates success; frame is the image data. If frame is None, the video has ended.
# 
# frame = cv.resize(frame, ...): 
# - Resizes the frame to half its original size. This improves processing speed.
# 
# gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY): 
# - Converts the frame to grayscale. Grayscale simplifies the processing flow.
# 
# blur = cv.GaussianBlur(gray, (3,3), 0): 
# - Applies Gaussian blurring to the grayscale frame. This reduces noise and small details.
# 
# *-----------Background Frame Handling (First Frame Only):-----------*
# 
# if background is None: : 
# - Checks if the background variable is empty (this happens only on the iteration of the while loop).
# 
# background = blur: 
# - Sets the first blurred, grayscale frame as the background reference.
# 
# *-----------Frame Differencing and Thresholding:-----------*
# 
# difference = cv.absdiff(background, blur): 
# - Calculates the absolute difference between the background frame and the current blurred frame. This highlights areas where pixels have changed.
# 
# kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)): 
# - Creates a small elliptical kernel for morphological operations.
# 
# dilation = cv.dilate(difference, kernel): 
# - Performs dilation on the difference image. Dilation expands the white regions (areas of change), making them into large white blobs.
# 
# fgmask = cv.threshold(dilation, 12, 255, cv.THRESH_BINARY)[1]: 
# - Applies a threshold to the dilated image. Pixels above the threshold (12) become white (255), and pixels below become black (0). This creates a binary mask (fgmask) where white represents foreground (motion).
# 
# *-----------Region of Interest (ROI) Definition:-----------*
# 
# height, width, _ = frame.shape: 
# - Gets the dimensions of the frame.
# 
# bX, bY, bX2, bY2: 
# - Defines the coordinates of a rectangular ROI. This restricts car detection to a specific area of the frame (the main street).
# 
# cv.line(frame, (0, bY), (bX2, bY), (0,0,255), 1): 
# - Draws a red line on the original frame to visually indicate the area of the Main Street.
# 
# *-----------Contour Detection and Filtering:-----------*
# 
# contours, _ = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE): 
# - Finds contours (outlines) in the binary foreground mask (fgmask). cv.RETR_EXTERNAL retrieves only the outer contours, and cv.CHAIN_APPROX_SIMPLE compresses the contour data.
# 
# for contour in contours: : 
# - Iterates through each detected contour.
# 
# if min_contour_area < cv.contourArea(contour) < (height * width)*0.5: : 
# - Filters contours based on their area. Only contours within the specified size range are considered potential cars.
# 
# *-----------Centroid Calculation and Bounding Box:-----------*
# 
# M = cv.moments(contour): 
# - Calculates the moments of the contour. Moments are used to find the centroid.
# 
# if M["m00"] != 0: : 
# - Avoids division by zero if the contour has no area.
# 
# cX = int(M["m10"] / M["m00"]), cY = int(M["m01"] / M["m00"]): 
# - Calculates the centroid (center) coordinates of the contour.
# 
# x, y, w, h = cv.boundingRect(contour): 
# - Gets the bounding rectangle coordinates (top-left corner x and y, width w, height h) of the contour.
# 
# *-----------ROI Check and Drawing:-----------*
# 
# if bX < cX < bX2 and bY < cY < bY2: : 
# - Checks if the centroid of the contour is within the Main Street.
# 
# cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1): 
# - If the contour is within the Main Street and meets the size criteria, a green rectangle is drawn around it on the original frame.
# 
# *-----------Display and Exit:-----------*
# 
# cv.imshow('Bounding Box Frame', frame), cv.imshow('frame differenced', fgmask): 
# - Displays the processed frame (with rectangles) and the computer's vision.
# 
# if keyboard == ord('q') or keyboard == 27: : 
# - Checks if the 'q' key or the Escape key was pressed, and breaks the loop if so to then exit the video.
# 
# *-----------Cleanup:-----------*
# 
# cap.release(): 
# - Releases the video capture object.
# 
# cv.destroyAllWindows(): 
# - Closes all OpenCV windows.

# %%
videos = ['Traffic_Laramie_2.mp4','Traffic_Laramie_1.mp4']
for video in videos:
    print(video)
    cap = cv.VideoCapture(video)
    fps = cap.get(cv.CAP_PROP_FPS)
    #to find the number "cars per minute"
    totalNoFrames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    durationInMinute = (totalNoFrames // fps) / 60

    if not cap.isOpened():
        print("Error opening video stream")
        exit()
    
    #generate background subtraction, setting up its parameter
    #500 is used because the enrivonment (from cars appearing in the 4-way junctions from time to time) 
    #is relatively static, but dynamic enough since. 
    #30 is used because we wants to reduce the "noise" (aka white dots) appeared on the screen
    fgbg = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=30,detectShadows=True)
    min_contour_area = 750
    min_centroid_distance = 45

    cars = {}
    counter= 0
    car_counter = 0

    print_once = True

    while True:
        ret, frame = cap.read()
        if frame is None:
            break

        frame = cv.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))

        fgmask = fgbg.apply(frame)

        #for task 2, the area of interest will be limited to the centre of the 4-way junction.
        # this will help to get the limit of the area, 
        # considering that the area of focus is at the bottom right corner of the screen
        #       bY
        # bX ----|---- width
        #       height
        height, width, _ = frame.shape
        bX = int(width*(1/2))
        bY = int(height*(5/11))

        contours, _ = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            #due to lesser noise from the background subtraction's threshhold = 30, 
            # theres no need to set a max dimension, 
            # the findContour() wont generate a contour for all of the noises around the whole screen 
            if min_contour_area < cv.contourArea(contour):
                x, y, w, h = cv.boundingRect(contour)
                M = cv.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    #if the detected car is within the centre of the 4-way junction
                    if bX< cX < width and bY < cY < height:

                        new_car = True
                        best_distance = float('inf')
                        best_match = None

                        #to find where did the car appear from.
                        # this helps to identify which car moved to the right,
                        # and wont mistakenly count cars that drove in from the right
                        initial = min(cX - bX, width - cX, cY - bY, height - cY)
                        if initial == cX - bX:
                            origin = "left"
                        elif initial == cY - bY:
                            origin = "top"
                        elif initial == width - cX:
                            origin = "right"
                        elif initial == height - cY:
                            origin = "bot"

                        #this will comapre with all contours considered to be a car in the dict
                        #if centroids are close to each other, the this mean the new contour is for the old car that is moving
                        # also, this will allows us to only select closest centroids and use it to update the for the current car
                        for car_key, car in cars.items():
                            distance = (car["cX"] - cX)**2 + (car["cY"] - cY)**2
                            if distance <= min_centroid_distance**2 and distance < best_distance:
                                best_distance = distance
                                best_match = car_key
                                new_car = False

                        #if the current centroids are not near any othes in the dict, this must mean it is a new car
                        if new_car == True:
                            #add the contour coords, centrods, and extra details to help with contour management into the dict
                            counter +=1
                            cars[counter] = {"x":x,"y":y,"w":w,"h":h,"cX":cX,"cY":cY,"counted":False,"life":10,"origin":origin}
                        else:
                            #if it is not a new car, this is where its new contour is updated into
                            cars[best_match].update({
                            "x": x,
                            "y": y,
                            "w": w,
                            "h": h,
                            "cX": cX,
                            "cY": cY,
                            "life":10
                    })
                            
        
        cars_to_remove = []
        for car_key,car_value in cars.items():
            #this is set up so that the car wont be immidietly marked for deletion when it appeared, 
            #cause it contour will be touching the edge of the border
            if car_value["origin"] == "left":
                if car_value['cX'] >= width or car_value['cY'] <= bY or car_value['cY'] >= height or car_value['life'] <= 0:
                    cars_to_remove.append(car_key)
            elif car_value["origin"] == "top":
                if car_value['cX']  <= bX or car_value['cX'] >= width  or car_value['cY'] >= height or car_value['life'] <= 0:
                    cars_to_remove.append(car_key)
            elif car_value["origin"] == "right":
                if car_value['cX']  <= bX or car_value['cY'] <= bY or car_value['cY'] >= height or car_value['life'] <= 0:
                    cars_to_remove.append(car_key)
            elif car_value["origin"] == "bot":
                if car_value['cX']  <= bX or car_value['cX'] >= width or car_value['cY'] <= bY or  car_value['life'] <= 0:
                    cars_to_remove.append(car_key)
            
            #display the contour + car number + where it came from
            cv.putText(frame, str(car_key)+"|"+car_value["origin"], (car_value['x'],car_value['y']), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 1, cv.LINE_AA)
            cv.rectangle(frame, (car_value["x"], car_value["y"]), (car_value["x"] +car_value["w"], car_value["y"] +car_value["h"]), (0, 0, 255), 1)
    
            #since we are only focusing on the centre of the 4-way junctions, cars all 99.999% always in motions.
            #so when a car does not have its contour updated for a duration, 
            #it will be considered that it have left the screen and removed
            car_value['life']-=1
            if car_value['life'] <= 0:
                #checks if the removed car is moving toward the city centre
                #this is done by checking if the car is not from the city centre (left)
                #and checks if the distance between the car's centroid to the left border is the shortest among others
                #it is compared with the car's centroid to the bottom/right/top borders
                #if the car is not from the left + the contour is closest to the left, +1 to the counter
                if car_value["origin"] != "left":
                    if abs(car_value['cX'] - bX) == min(abs(width - car_value['cX']),abs(car_value['cX'] - bX),abs(car_value['cY'] - bY),abs(height - car_value['cY'])) and car_value["counted"] == False:
                        car_counter+=1
                        car_value["counted"] = True

        #since i am using a dictionary to keep tracks of all cars present, 
        # this is the best way to remove the dict content
        for car_key in cars_to_remove:
            del cars[car_key]
        
        text_position = (10, 30) # top left of screen
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        color = (255, 255, 255)
        thickness = 1 
        #so we can know what are deing detected
        text = "Cars: " + str(car_counter) + "||" + "width: "+ str(width) + ",height: "+str(height)
        cv.putText(frame, text, text_position, font, font_scale, color, thickness, cv.LINE_AA)
        cv.putText(frame, str(cars), (10,50), font, 0.30, (0,0,255), thickness, cv.LINE_AA)

        # display the computer's vision of the video
        cv.imshow('FG MASK Frame',fgmask)
        # display the human view of the video + the counter + contours
        cv.imshow('Bounding Box Frame',frame)

        keyboard = cv.waitKey(30)
        if keyboard == ord('q') or keyboard == 27:
            break

    cap.release()
    cv.destroyAllWindows()
    #for the task 2's table that is to be filled
    print("car that moves to the left: "+ str(car_counter))
    print("cars per minutes detected: "+ str(counter/durationInMinute))

# %% [markdown]
# Task 2  
# *-----------Initialization (per video):-----------*
# 
# videos = ['Traffic_Laramie_2.mp4','Traffic_Laramie_1.mp4']: 
# - A list of video files to process.
# 
# for video in videos: : 
# - Loops through each video file.
# 
# cap = cv.VideoCapture(video): 
# - Opens the current video file.
# 
# fps = cap.get(cv.CAP_PROP_FPS), totalNoFrames = cap.get(cv.CAP_PROP_FRAME_COUNT), durationInMinute = ...: 
# - Calculates video properties (frames per second, total frames, duration in minutes) for later use in calculating cars per minute.
# 
# fgbg = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=30, detectShadows=True):
# - Creates a background subtractor object using the MOG2 algorithm. history=500 means it uses 500 frames to build the background model. varThreshold=30 sets the threshold for foreground detection. detectShadows=True enables shadow detection.
# 
# min_contour_area = 750, min_centroid_distance = 45: 
# - Defines thresholds for contour area and centroid distance.
# 
# cars = {}, counter = 0, car_counter = 0: 
# - Initializes variables to store car data, a general counter, and a counter specifically for cars moving left.
# 
# *-----------Main Processing Loop (while True):-----------*
# 
# ret, frame = cap.read(): 
# - Reads a frame from the video.
# 
# frame = cv.resize(frame, ...):
# - Resizes the frame for faster processing.
# 
# fgmask = fgbg.apply(frame): 
# - Applies the background subtraction to the current frame. This produces a foreground mask (fgmask) where white pixels represent cars.
# 
# *-----------Region of Interest (ROI) Definition:-----------*
# 
# height, width, _ = frame.shape: 
# - Gets frame dimensions.
# 
# bX = int(width*(1/2)), bY = int(height*(5/11)): 
# - Defines the ROI, focusing on the center of the four-way junction. This is different from Task 1's ROI.
# 
# *-----------Contour Detection and Filtering:-----------*
# 
# contours, _ = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE): 
# - Finds contours in the foreground mask.
# 
# for contour in contours: : 
# - Iterates through each contour.
# 
# if min_contour_area < cv.contourArea(contour): : 
# - Filters contours based on minimum area.
# 
# *-----------Centroid Calculation and Bounding Box:-----------*
# 
# Calculates the centroid (cX, cY) and bounding box (x, y, w, h) of the contour (same as in Task 1).
# 
# *-----------ROI and New Car Check:-----------*
# 
# if bX < cX < width and bY < cY < height: : 
# - Checks if the centroid is within the ROI.
# 
# new_car = True, best_distance = float('inf'), best_match = None: 
# - Initializes variables for tracking cars.
# 
# initial = ..., origin = ...: 
# - Determines the car's origin (left, top, right, bot) based on its initial position relative to the ROI. This is crucial for directional counting.
# 
# *-----------Car Tracking and Matching:-----------*
# 
# for car_key, car in cars.items(): : 
# - Iterates through the cars dictionary, which stores information about previously detected cars.
# 
# distance = ...: 
# - Calculates the squared Euclidean distance between the current contour's centroid and the centroid of a previously detected car.
# 
# if distance <= min_centroid_distance**2 and distance < best_distance: : 
# - Checks if the current contour is close enough to a previous car and if it's the closest match found so far.
# 
# Updates best_distance, best_match, and sets new_car = False if a close match is found.
# 
# *-----------New Car Handling:-----------*
# 
# if new_car == True:: 
# - If no close match is found, this is considered a new car.
# 
# counter += 1: 
# - Increments the general counter.
# 
# cars[counter] = ...: 
# - Adds a new entry to the cars dictionary, storing the car's information (coordinates, centroid, origin, etc.). life is initialized to 10, representing the car's "lifespan" before it's considered to have left the scene. counted is initialized to False, indicating whether this car has been counted towards the "cars moving left" total.
# 
# *-----------Existing Car Update:-----------*
# 
# else: : 
# - If a close match was found (not a new car).
# 
# cars[best_match].update(...): 
# - Updates the information for the existing car in the cars dictionary with the new contour data. life is reset to 10.
# 
# *-----------Car Removal and Counting:-----------*
# 
# cars_to_remove = []: 
# - Create a list for cars' dict key to be removed
# 
# for car_key, car_value in cars.items(): : 
# - Iterates through the cars dictionary.
# 
# if car_value["origin"] == ...: 
# - A series of if and elif statements that determines which border of the ROI the car originated from. It checks if the car has moved outside the ROI or if its life has reached 0. If either condition is true based on the origin, the car's key is added to cars_to_remove.
# 
# cv.putText(...), cv.rectangle(...): 
# - Draws the car's ID, origin, and a bounding box on the frame (for visualization).
# 
# car_value['life'] -= 1: 
# - Decrements the car's life counter.
# 
# if car_value['life'] <= 0: : 
# - Checks if the car's life has reached zero (meaning it hasn't been detected recently).
# 
# - The nested if statements inside check if the car is moving towards the left (into the city center) and hasn't been counted yet (car_value["counted"] == False). If both are true, car_counter is incremented, and car_value["counted"] is set to True.
# 
# for car_key in cars_to_remove:
#     del cars[car_key]: 
# - Remove cars that has left the ROI
# 
# *-----------Display and Exit:-----------*
# 
# cv.putText(frame, ...) (multiple calls): 
# - Adds text to the frame, displaying the number of cars counted, frame dimensions, and the contents of the cars dictionary (for debugging).
# 
# cv.imshow(...): 
# - Displays the foreground mask (computer vision) and the processed frame.
# 
# Handles key presses for exit (same as in Task 1).
# 
# *-----------Cleanup and Results (per video):-----------*
# 
# cap.release(), cv.destroyAllWindows(): Releases resources.
# 
# print("car that moves to the left: " + str(car_counter)): 
# - Prints the final count of cars moving left.
# 
# print("cars per minutes detected: " + str(counter/durationInMinute)): 
# - Prints the cars per minute.

# %% [markdown]
# REFERENCE:
# 
# base code + example : 
# 
# ProgrammingKnowledge (2019) Contours - Ep 16 - OpenCV with Python for Image and Video Analysis. Available at: https://www.youtube.com/watch?v=eZ2kDurOodI (Accessed: 23 October 2024).
# 
# mog2 paratemeters: 
# 
# OpenCV (2024) Motion Analysis. Available at: https://docs.opencv.org/4.x/de/de1/group__video__motion.html#ga818a6d66b725549d3709aa4cfda3f301 (Accessed: 23 October 2024).
# 
# background subtraction with rectangle example: 
# 
# OpenCV (2024) Background Subtraction. Available at: https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html (Accessed: 23 October 2024).
# 
# contour:
# 
# LearnOpenCV (2023) Contours - Ep 14 - OpenCV with Python for Image and Video Analysis. Available at: https://www.youtube.com/watch?v=FKIn_UIsXy4 (Accessed: 23 October 2024).
# 
# contour's features:
# 
# OpenCV (2024) Contours Features. Available at: https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html (Accessed: 23 October 2024).
# 
# get the video's duration: 
# 
# Tutorialspoint (2023) Get Video Duration using OpenCV Python. Available at: https://www.tutorialspoint.com/get-video-duration-using-opencv-python (Accessed: 23 October 2024).
# 


