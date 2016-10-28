#!/usr/bin/env python
# encoding: utf-8


from skimage.transform import pyramid_gaussian
from skimage.io import imread
from skimage.feature import hog
from sklearn.externals import joblib
import cv2
import os
import pandas as pd
import ConfigParser as cp
import json
from config import RES_FOLDER



class CarDetector(object):
    def __init__(self, model_path="./data/models/car_LIN_SVM.model"):
        config = cp.RawConfigParser()
        config.read('./data/config.cfg')

        self.model_path = model_path
        self.clf = joblib.load(model_path)
        self.image_path = None
        self.test_dir = None
        self.annotation_path = ""
        self.min_wdw_sz = json.loads(config.get("hog", "min_wdw_sz"))
        self.step_size = json.loads(config.get("hog", "step_size"))
        self.orientations = config.getint("hog", "orientations")
        self.pixels_per_cell = json.loads(config.get("hog", "pixels_per_cell"))
        self.cells_per_block = json.loads(config.get("hog", "cells_per_block"))
        self.downscale = config.getfloat("nms", "downscale")
        self.visualize = config.getboolean("hog", "visualize")
        self.normalize = config.getboolean("hog", "normalize")
        self.threshold = config.getfloat("nms", "threshold")
        self.output_dir = RES_FOLDER

    def run(self, params, params_path):
        self.params = params
        assert type(self.params) == str
        if params == 'image':
            org_img, nms_img, nms_num = self._process_one_img(params_path, self.clf)
            cv2.imwrite(os.path.join(self.output_dir, 'org_' + os.path.split(params_path)[1]), org_img)
            cv2.imwrite(os.path.join(self.output_dir, os.path.split(params_path)[1]), nms_img)
            nms_fn = os.path.split(params_path)[1]
            return nms_num,nms_fn
        elif params == 'dir':
            self._process_dir_img(params_path, self.clf)
        else:
            raise Exception("unsupported parameters : {}".format(params))


    def _overlapping_area(self, detection_1, detection_2):
        '''
        Function to calculate overlapping area'si
        `detection_1` and `detection_2` are 2 detections whose area
        of overlap needs to be found out.
        Each detection is list in the format ->
        [x-top-left, y-top-left, confidence-of-detections, width-of-detection, height-of-detection]
        The function returns a value between 0 and 1,
        which represents the area of overlap.
        0 is no overlap and 1 is complete overlap.
        Area calculated from ->
        http://math.stackexchange.com/questions/99565/simplest-way-to-calculate-the-intersect-area-of-two-rectangles
        '''
        # Calculate the x-y co-ordinates of the
        # rectangles
        x1_tl = detection_1[0]
        x2_tl = detection_2[0]
        x1_br = detection_1[0] + detection_1[3]
        x2_br = detection_2[0] + detection_2[3]
        y1_tl = detection_1[1]
        y2_tl = detection_2[1]
        y1_br = detection_1[1] + detection_1[4]
        y2_br = detection_2[1] + detection_2[4]
        # Calculate the overlapping Area
        x_overlap = max(0, min(x1_br, x2_br) - max(x1_tl, x2_tl))
        y_overlap = max(0, min(y1_br, y2_br) - max(y1_tl, y2_tl))
        overlap_area = x_overlap * y_overlap
        area_1 = detection_1[3] * detection_2[4]
        area_2 = detection_2[3] * detection_2[4]
        total_area = area_1 + area_2 - overlap_area
        return overlap_area / float(total_area)

    def _nms(self, detections, threshold=.5):
        '''
        This function performs Non-Maxima Suppression.
        `detections` consists of a list of detections.
        Each detection is in the format ->
        [x-top-left, y-top-left, confidence-of-detections, width-of-detection, height-of-detection]
        If the area of overlap is greater than the `threshold`,
        the area with the lower confidence score is removed.
        The output is a list of detections.
        '''
        if len(detections) == 0:
            return []
        # Sort the detections based on confidence score
        detections = sorted(detections, key=lambda detections: detections[2],
                            reverse=True)
        # Unique detections will be appended to this list
        new_detections = []
        # Append the first detection
        new_detections.append(detections[0])
        # Remove the detection from the original list
        del detections[0]
        # For each detection, calculate the overlapping area
        # and if area of overlap is less than the threshold set
        # for the detections in `new_detections`, append the
        # detection to `new_detections`.
        # In either case, remove the detection from `detections` list.
        for index, detection in enumerate(detections):
            for new_detection in new_detections:
                if self._overlapping_area(detection, new_detection) > threshold:
                    del detections[index]
                    break
            else:
                new_detections.append(detection)
                del detections[index]
        return new_detections

    def _sliding_window(self, image, window_size, step_size):
        '''
        This function returns a patch of the input image `image` of size equal
        to `window_size`. The first image returned top-left co-ordinates (0, 0)
        and are increment in both x and y directions by the `step_size` supplied.
        So, the input parameters are -
        * `image` - Input Image
        * `window_size` - Size of Sliding Window
        * `step_size` - Incremented Size of Window

        The function returns a tuple -
        (x, y, im_window)
        where
        * x is the top-left x co-ordinate
        * y is the top-left y co-ordinate
        * im_window is the sliding window image
        '''
        for y in xrange(0, image.shape[0], step_size[1]):
            for x in xrange(0, image.shape[1], step_size[0]):
                yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

    def _process_one_img(self, im_path, clf):
        # Read the image
        im = imread(im_path, as_grey=True)
        # cv2.imshow('t',im)
        # cv2.waitKey(0)
        # List to store the detections
        detections = []
        # The current scale of the image
        scale = 0
        # Downscale the image and iterate
        for im_scaled in pyramid_gaussian(im, downscale=self.downscale):

            # This list contains detections at the current scale
            cd = []
            # If the width or height of the scaled image is less than
            # the width or height of the window, then end the iterations.
            if im_scaled.shape[0] < self.min_wdw_sz[1] or im_scaled.shape[1] < self.min_wdw_sz[0]:
                break
            for (x, y, im_window) in self._sliding_window(im_scaled, self.min_wdw_sz, self.step_size):
                if im_window.shape[0] != self.min_wdw_sz[1] or im_window.shape[1] != self.min_wdw_sz[0]:
                    continue
                # Calculate the HOG features
                fd = hog(im_window, self.orientations, self.pixels_per_cell, self.cells_per_block, self.visualize,
                         self.normalize)
                fd = fd.reshape(1, -1)
                pred = clf.predict(fd)
                if pred == 1:
                    # print  "Detection:: Location -> ({}, {})".format(x, y)
                    # print "Scale ->  {} | Confidence Score {} \n".format(scale, clf.decision_function(fd))
                    detections.append((x, y, clf.decision_function(fd),
                                       int(self.min_wdw_sz[0] * (self.downscale ** scale)),
                                       int(self.min_wdw_sz[1] * (self.downscale ** scale))))
                    cd.append(detections[-1])
                # If visualize is set to true, display the working
                # of the sliding window
                if self.visualize:
                    clone = im_scaled.copy()
                    for x1, y1, _, _, _ in cd:
                        # Draw the detections at this scale
                        cv2.rectangle(clone, (x1, y1), (x1 + im_window.shape[1], y1 +
                                                        im_window.shape[0]), (0, 0, 0), thickness=2)
                    cv2.rectangle(clone, (x, y), (x + im_window.shape[1], y +
                                                  im_window.shape[0]), (255, 255, 255), thickness=2)
                    cv2.imshow("Sliding Window in Progress", clone)
                    cv2.waitKey(30)
            # Move the the next scale
            scale += 1

        # Display the results before performing NMS
        org_res = im.copy()
        for (x_tl, y_tl, _, w, h) in detections:
            # Draw the detections
            cv2.rectangle(org_res, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 0), thickness=2)

        # Perform Non Maxima Suppression
        detections = self._nms(detections, self.threshold)

        nms_res = im.copy()
        # Display the results after performing NMS
        for (x_tl, y_tl, _, w, h) in detections:
            # Draw the detections
            cv2.rectangle(nms_res, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 0), thickness=2)
        # cv2.imshow("Final Detections after applying NMS", nms_res)
        # cv2.waitKey(0)
        return org_res, nms_res, len(detections)

    def _process_dir_img(self, params_path, clf):
        input_dir = params_path
        # process dir
        fl = os.listdir(input_dir)
        test_result = pd.DataFrame(columns=['img_name', 'truth', 'predict_cnt', 'rate'])
        test_result['img_name'] = pd.Series(fl)
        image_current = 0
        for fn in fl:
            im_path = os.path.join(input_dir, fn)
            print im_path
            org_img, nms_img, nms_num = self._process_one_img(im_path, clf)
            cv2.imwrite(os.path.join(self.output_dir, 'org_' + fn), org_img)
            cv2.imwrite(os.path.join(self.output_dir, 'nms_' + fn), nms_img)
            test_result.iloc[image_current, 2] = nms_num
            if self.annotation_path != "":
                temp = os.path.split(os.path.join(self.annotation_path, fn))
                fr = open(temp[0] + '/' + temp[1].split(".")[0] + '.txt')
                aonnotation = fr.read()
                real_count = aonnotation.count('Original label for object')
                test_result.iloc[image_current, 1] = real_count
                test_result.iloc[image_current, 3] = max(0, 1 - float(abs(real_count - nms_num)) / real_count)
            image_current += 1
        test_result.to_csv(self.output_dir + 'test_result.csv', header=True, index=False)
