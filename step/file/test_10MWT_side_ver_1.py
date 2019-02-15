import sys, os, math
import poselib
import cv2
import time
import argparse
import os
import numpy as np
import imutils
from RectangleDetector import RectangleDetector

parser = argparse.ArgumentParser(description='test_poselib run')
parser.add_argument('-v', '--video', type=str, help='path of input video', required=True)
parser.add_argument('-r', '--resize', type=str, default='1', help='resize input image 432x368 choose 1 / 656x368 choose 2 / 1312x736 choose 3')
parser.add_argument('-m', '--model', type=str, default='mobilenet_thin', help='choose model mobilenet_thin or cmu')
parser.add_argument('-o','--output', type=str, default='output_video/',help='path of output folder')
args = parser.parse_args()

#============================================
# load model and weights with class PoseModel
#============================================
pose_model = poselib.PoseModel(args.model, args.resize)

class PATEAi10MWTSIDE:
    def __init__(self, recDetector, model, video_path, output='./output/', resize=1):
        
        # set parameter for detect marker
        self.lower_range = np.array([150,100,50])
        self.upper_range = np.array([189,255,255])
        self.recDetector = recDetector
        self.left_marker = None
        self.right_marker = None

        # set model for detect bodypoints
        self.model = model
        
        # set parameter for detecting time
        self.start_time_part1 = None
        self.stop_time_part1 = None
        self.is_start_part1 = False
        self.is_stop_part1 = False

        self.start_time_part2 = None
        self.stop_time_part2 = None
        self.is_start_part2 = False
        self.is_stop_part2 = False

        # video parameter
        self.fps = 0
        self.total_frame = 0
        self.video_path = video_path
        self.output = output

        # set parameter for monitor and checking rule
        self.distance = []
        self.max_distance = 0
        self.init_max_distance = 0
        self.max_distance_list = []
        self.counted_max_distance_list = []
        self.right_first = []
        self.mean_ankle = 0

    def detect_marker(self, image):
        """
        detect the red marker in the image, function does following steps in image:
        - filter image with red roi
        - find the roi with red color
        - detect rectangle shape roi, rotation, width, height
        - filt out detected retangle with ratio between width and height in recDetector class
        for mor accurate detecting rectangle -> adjust threshold of ratio between width and height
        of rectangle in recDetector
            :param image: input image
        return:
            :coords_marker: list of 4 corner of all detected rectangle in image
        TODO:
            try different method for improving the accuracy of rectangle detecting
        """   
        # convert color space to hsv
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # finding the roi with red color in the image
        red = cv2.inRange(hsv_img, self.lower_range, self.upper_range)
        # smooth image using gaussian with 5x5 kernel to suppress noise
        red_blurred = cv2.GaussianBlur(red, (5, 5), 0)
        # threshold, dilated roi with 5x5 kernel        
        kernel = np.ones((5,5), "uint8")
        red_thresh = cv2.threshold(red_blurred, 60, 255, cv2.THRESH_BINARY)[1]
        red_dilate = cv2.dilate(red_thresh, kernel)
        # find contours in the thresholded image then find bbox of roi
        cnts = cv2.findContours(red_dilate, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        # refit the contours arcording to opencv version
        cnts = imutils.grab_contours(cnts)
        # choose the bbox which is rectangle using rectangle detector class 
        # and return 4 corner of rectangle together with centroid position,
        # size of contour
        marker = []
        centroid = []
        size = []
        corners = []
        for cnt in cnts:
            detected = self.recDetector.detect(cnt)
            if detected[0]:
                centroid.append(detected[1])
                size.append(detected[2])
                corners.append(detected[3])
        # only return marker with 2 contour
        # if there are 3 contour detected, select 2 largest contour base on size
        if len(centroid) == 1:
            return marker
        else:
            if len(centroid) > 2:
                sort_idx = np.argsort(size)[-1:-3]
                centroid = np.array(centroid)[sort_idx]
                size = np.array(size)[sort_idx]
                corners = np.array(corners)[sort_idx]
                # return data with contour group
            elif len(centroid) == 2:
                centroid = np.array(centroid)
                size = np.array(size)
                corners = np.array(corners)
            for i in range(len(centroid)):
                marker.append([centroid[i], size[i], corners[i]])
        return marker

    def set_marker(self, new_marker):
        """
        compare new marker with current marker, if new marker is larger than
        current marker, set current marker is new marker
            :param new_marker: markers which are detected in the current frame
        """ 
        if new_marker:
            left_marker = []
            right_marker = []
            # get max and min x in marker 1
            # x_max1 = np.max(new_marker[0][2], axis=0)[0]
            x_min1 = np.min(new_marker[0][2], axis=0)[0]
            # get max and min x in marker 2
            x_max2 = np.max(new_marker[1][2], axis=0)[0]
            # x_min2 = np.min(new_marker[1][2], axis=0)[0]
            # check which marker is left and right
            if x_min1 > x_max2:
                left_marker = new_marker[1]
                right_marker = new_marker[0]
            else:
                left_marker = new_marker[0]
                right_marker = new_marker[1]
            # compare with current marker
            # if not detected yet, set current marker
            if self.left_marker is None:
                self.left_marker = left_marker
                self.right_marker = right_marker
            # if new marker is larger than current marker, use new marker
            if self.left_marker[1] < left_marker[1]:
                self.left_marker = left_marker
            if self.right_marker[1] < right_marker[1]:
                self.right_marker = right_marker

    def computeDistance2Vector(self, input_dict):
        """
        if bodypoint sets contain 3 point [neck, hip, ankle] then compute the length of 2 vector below,
        either left side or right side
        vector 1: neck point - hip point
        vector 2: hip point - ankle point
            :param input_dict: bodypoint set contain keypoints for computing
        return:
            :distance: 0 if not contain required point, or sum of 2 vector's norm 
        """
        distance = 0
        # right side
        if 1 in input_dict and 8 in input_dict and 9 in input_dict:
            v0 = (input_dict[1][0] - input_dict[8][0], input_dict[1][1] - input_dict[8][1])
            v1 = (input_dict[9][0] - input_dict[8][0], input_dict[9][1] - input_dict[8][1])
            distance = np.linalg.norm(v0) + np.linalg.norm(v1) 
        # left side
        elif 1 in input_dict and 11 in input_dict and 12 in input_dict:
            v0 = (input_dict[1][0] - input_dict[11][0], input_dict[1][1] - input_dict[11][1])
            v1 = (input_dict[12][0] - input_dict[11][0], input_dict[12][1] - input_dict[11][1])
            distance = np.linalg.norm(v0) + np.linalg.norm(v1) 
        return distance

    def parse_result_to_dict(self, results):
        """
        get detected body keypoints set in list result that has maximum length in image
            :param results: list of detected body keypoints sets
        return:
            :None if the results list is None or
            :keypoint_dict: dict type of keypoints set
        """   
        if results is not None:
            max_distance_position = 0
            bodypart_dicts = []
            # transform human class to dictionary type
            for i in range(len(results)):
                id_kp = results[i].body_parts.keys()
                keypoint_dict = {}
                for key in id_kp:
                    body_part = results[i].body_parts[key]
                    keypoint_dict[key] = [body_part.x, body_part.y]
                bodypart_dicts.append(keypoint_dict)
            # get the set has maximum length
            for i in range(len(bodypart_dicts)):
                if self.computeDistance2Vector(bodypart_dicts[i]) > \
                   self.computeDistance2Vector(bodypart_dicts[max_distance_position]):
                    max_distance_position = i
            return bodypart_dicts[max_distance_position]
        else:
            return None

    def rescale_keypoint_dict(self, image, keypoint_dict):
        """
        rescale the keypoint value from normalized value to image pixel value
            :param image: input image
            :param keypoint_dict: dict type keypoints set
        return:
            :None if keypoint_dict is None
            :keypoint_dict: dict type keypoints set with location scale to
                            pixel value
        """   
        if keypoint_dict is not None:
            image_h, image_w = image.shape[:2]
            for key in keypoint_dict:
                keypoint_dict[key][0] = int(keypoint_dict[key][0]*image_w)
                keypoint_dict[key][1] = int(keypoint_dict[key][1]*image_h)
            return keypoint_dict
        else:
            return None
            
    def index_to_time(self, index, fps):
        """
        convert frame index to time
            :param index: frame index
            :param fps: camera fps
        return:
            :time value in type string with hour:minute:second format
        """
        seconds = int(index / fps)
        return time.strftime("%H:%M:%S", time.gmtime(seconds))

    def computeDistance2Ankle(self, RAnkle, LAnkle):
        """
        compute absolute distance between two ankle
            :param RAnkle: right ankle point (10)
            :param LAnkle: left ankle point (13)
        """   
        return np.abs(RAnkle[0] - LAnkle[0])

    def detectStarttimepart1(self, lAnkle, rAnkle, frame_idx):
        if lAnkle[0] <= self.right_marker[0][0] or \
            rAnkle[0] <= self.right_marker[0][0]:
            print('START PART 1')
            self.is_start_part1 = True
            self.start_time_part1 = self.index_to_time(frame_idx, self.fps)

    def detectStoptimepart1(self, lAnkle, rAnkle, frame_idx):
        if lAnkle[0] <= self.left_marker[0][0] and \
            rAnkle[0] <= self.left_marker[0][0]:
            print('STOP PART 1')
            self.is_stop_part1 = True
            self.stop_time_part1 = self.index_to_time(frame_idx, self.fps)

    def detectStarttimepart2(self, lAnkle, rAnkle, frame_idx):
        if lAnkle[0] >= self.left_marker[0][0] or \
            rAnkle[0] >= self.left_marker[0][0]:
            if frame_idx > (self.total_frame/2)+ self.fps:
                print('START PART 2')
                self.is_start_part2 = True
                self.start_time_part2 = self.index_to_time(frame_idx, self.fps)

    def detectStoptimepart2(self, lAnkle, rAnkle, frame_idx):
        if lAnkle[0] >= self.right_marker[0][0] and \
            rAnkle[0] >= self.right_marker[0][0]:
            print('STOP PART 2')
            self.is_stop_part2 = True
            self.stop_time_part2 = self.index_to_time(frame_idx, self.fps)

    def detectTime(self, start_time, stop_time):
        def get_sec(time_str):
            h, m, s = time_str.split(':')
            return int(h) * 3600 + int(m) * 60 + int(s)
        if start_time is not None and stop_time is not None:
            return np.abs(get_sec(stop_time) - get_sec(start_time))
        else:
            return None

    def check_change_leg(self, mean_ankle_dis, distance, threshold, scale_1, scale_2):
        if mean_ankle_dis>(threshold*scale_1) and distance>=(threshold*scale_2):
            return True
        else:
            return False

    def detectSteps(self, keypoints_dict, frame_idx):
        # camera in right side of human: RAnkle(10), LAnkle(13), LHip(11)
        # camera in left side of human: RAnkle(10), LAnkle(13), RHip(8)
        keys = set(keypoints_dict.keys())
        ankle_key = set([10, 13])
        if ankle_key.issubset(keys):
            # compute distance between two ankle and append to monitor list
            rAnkle = keypoints_dict[10]
            lAnkle = keypoints_dict[13]
            distance = self.computeDistance2Ankle(rAnkle, lAnkle)
            self.distance.append(distance)
            if not self.is_start_part1 and self.right_marker is not None:
                self.detectStarttimepart1(lAnkle, rAnkle, frame_idx)
            if frame_idx == 0:
                self.mean_ankle = (rAnkle[0]+lAnkle[0])/2
                self.init_max_distance = distance
            if distance > self.max_distance:
                self.max_distance = distance
                self.frame_idx = frame_idx
            # if distance is larger than a threshold (1/3 of last max_distance)
            # then check which leg is front and save the status
            mean_ankle = (rAnkle[0]+lAnkle[0])/2
            mean_ankle_dis = np.abs(mean_ankle-self.mean_ankle) 
            if not self.max_distance_list:
                threshold = self.init_max_distance
                scale_1 = 2
                scale_2 = 2
            else:
                threshold = np.max(self.max_distance_list)
                scale_1 = 0.3
                scale_2 = 0.4
            if rAnkle[0] > lAnkle[0]:
                # TODO: how about check distance 1st, if above threshold then check bigger 
                # leg, if same as previous  then do not thing else change and log
                if not self.right_first or self.right_first[-1]:
                    self.right_first.append(True)
                elif not self.right_first[-1]:
                    # condition ???  check leg or something
                    check = self.check_change_leg(mean_ankle_dis, distance, threshold,
                                                  scale_1, scale_2)
                    if check:
                        self.right_first.append(True)
                        self.mean_ankle = mean_ankle
                    else:
                        self.right_first.append(False)
            else:
                if not self.right_first or not self.right_first[-1]:
                    self.right_first.append(False)
                elif self.right_first[-1]:
                    # condition ???  check leg or something
                    check = self.check_change_leg(mean_ankle_dis, distance, threshold,
                                                  scale_1, scale_2)
                    if check:
                        self.right_first.append(False)
                        self.mean_ankle = mean_ankle
                    else:
                        self.right_first.append(True)
            # if status of right_first turn from True to False or vice versa
            # save current max_distance and reset
            if len(self.right_first) > 1:
                if self.right_first[-1] != self.right_first[-2]:
                    #if frame_idx > 1925:
                    #    self.true.append(self.max_mean_distance)
                    if (self.is_start_part1 is True and self.is_stop_part1 is False) or \
                        (self.is_start_part2 is True and self.is_stop_part2 is False):
                        self.counted_max_distance_list.append(self.max_distance)
                    self.log_msg(4, 'Change at FrameID: '+str(frame_idx)+'\n')
                    self.log_msg(4, '  Threshold:'+str(threshold)+'\n')
                    self.log_msg(4, '  Scale1: '+str(scale_1)+' Scale2: '+str(scale_2)+'\n')
                    self.log_msg(4, '  Mean ankle distance: '+str(mean_ankle_dis)+'\n')
                    self.log_msg(4, '  Ankle distance: '+str(distance)+'\n') 
                    self.max_distance_list.append(self.max_distance)
                    self.log_msg(3, 'FrameID: ' + str(self.frame_idx))
                    self.log_msg(3, ' '+str(self.max_distance)+'\n')
                    self.max_distance = 0
            self.log_msg(5, 'Frame ID:'+str(frame_idx)+' - Status:'+str(self.right_first[-1])+'\n')
            if not self.is_stop_part1 and self.left_marker is not None:
                self.detectStoptimepart1(lAnkle, rAnkle, frame_idx)
            if not self.is_start_part2 and self.left_marker is not None and self.is_stop_part1 is True:
                self.detectStarttimepart2(lAnkle, rAnkle, frame_idx)
            if not self.is_stop_part2 and self.right_marker is not None and self.is_stop_part1 is True:
                self.detectStoptimepart2(lAnkle, rAnkle, frame_idx)
        else:
            # poselib does not detect both ankle, can't compute distance, assign -1 and return
            distance = 0
            self.distance.append(distance)
        return distance

    def log_msg(self, type, msg=''):
        """
        logging function for debug and analysis
            :param type: type of information
                1:detected bodypoints
                2:distance between ankle each frame
                3:maximum distance each step cycle
                4:frame id that ankle change status
                5:status of front left at each frame
            :param msg='': logged data
        """  
        if type == 1:
            with open("bodypoints.txt", "a") as myfile:
                myfile.write(msg)
                # print(msg)
        if type == 2:
            with open("distance.txt", "a") as myfile:
                myfile.write(msg )
                # print(msg)
        if type == 3:
            with open("max_distance.txt", "a") as myfile:
                myfile.write(msg)
                # print(msg)
        if type == 4:
            with open("change_frame.txt", "a") as myfile:
                myfile.write(msg)
                # print(msg)
        if type == 5:
            with open("frame_status.txt", "a") as myfile:
                myfile.write(msg)
                # print(msg)

    def processing(self):
        cap = cv2.VideoCapture(self.video_path)
        if cap.isOpened() is False:
            print("Error opening video stream or file")
        file_name = os.path.basename(self.video_path)
        if not os.path.exists(self.output):
            os.makedirs(self.output)
        # get spec information of video
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        out = cv2.VideoWriter(self.output + file_name,
                              cv2.VideoWriter_fourcc(*'MP4V'), self.fps,
                              (frame_width,frame_height))
        # read frame by frame
        pos = 0
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                self.log_msg(1, 'FrameID: ' + str(pos))
                self.log_msg(2, 'FrameID: ' + str(pos))
                # detect marker in the first 30 frame:
                if pos < 30:
                    new_markers = self.detect_marker(frame)
                    self.set_marker(new_markers)
                # detect keypoint sets, get one set, convert to dict type
                # and rescale to image original pixel value
                body_keypoints = poselib.pose_detect(frame, self.model)
                keypoints_dict = self.parse_result_to_dict(body_keypoints)
                keypoints_dict = self.rescale_keypoint_dict(frame, keypoints_dict)
                # check step in frame
                if keypoints_dict is not None:
                    self.log_msg(1, ' - Bodypoint: '+str(keypoints_dict)+'\n')
                    distance = self.detectSteps(keypoints_dict, pos)
                    self.log_msg(2, ' - Distance: '+str(distance)+'\n')
                    poselib.draw_keypoints(frame, body_keypoints)
                    # for testing using self.max_distance_list for count from the 
                    # beginning of video
                    # if want to count when detect start time, use
                    # self.counted_max_distance_list
                    if self.counted_max_distance_list:
                        # threshold = np.max(self.max_distance_list)*0.6
                        # true_count = np.sum(np.array(self.max_distance_list)>threshold)
                        threshold = np.max(self.counted_max_distance_list)*0.6
                        true_count = np.sum(np.array(self.counted_max_distance_list)>threshold)
                    else:
                        true_count = 0
                    cv2.putText(frame, str(true_count)+'  '+str(distance), 
                                (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                (255,0,0))
                    # draw marker
                    if self.left_marker is not None:
                        cv2.drawContours(frame,[self.left_marker[2]], 0,(0,0,255),2)
                        point = self.left_marker[2][0]
                        cv2.putText(frame, 'left', (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0))
                        cv2.drawContours(frame,[self.right_marker[2]], 0,(0,0,255),2)
                        point = self.right_marker[2][0]
                        cv2.putText(frame, 'right', (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0))
                else:
                    cv2.putText(frame, str(count)+'  Not detect keypoints', 
                                (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                (255,0,0))
                # Write the frame into the file output
                out.write(frame)
                # uncomment following code for showing result frame by frame
                h,w,_ = frame.shape
                frame_rs = cv2.resize(frame,(int(w*0.6),int(h*0.6)))
                cv2.imshow('result', frame_rs)
                if cv2.waitKey(27) & 0xFF == ord('q'):
                    break
            else:
                break
            pos += 1
        cap.release()
        cv2.destroyAllWindows()
        return true_count

    def detectTest(self):
        """
        begin processing video and display result
        """
        count = self.processing()
        print('start time part 1: ', self.start_time_part1)
        print('stop time part 1: ', self.stop_time_part1)
        print('timing part 1: ', self.detectTime(self.start_time_part1, self.stop_time_part1))
        print('start time part 2: ', self.start_time_part2)
        print('stop time part 2: ', self.stop_time_part2)
        print('timing part 2: ', self.detectTime(self.start_time_part2, self.stop_time_part2))
        print('total stride: ', count)

# initialize rectangle detector with 2.5 ratio between 2 edge
recDetector = RectangleDetector(2.5)
obj = PATEAi10MWTSIDE(recDetector, pose_model, args.video, args.output, args.resize)
obj.detectTest()
