import cv2
import numpy as np

class RectangleDetector:
    """
    class for detect rectangle shape with user defined width and height ratio
    """
    def __init__(self, recEdgeRatio):
        """
        initialize detector with input ratio = long edge/short edge
        ex: 
        + if user want to detect rectange with long edge  twice short edge, input 2
        + if user want to detect square, input 1
            :param recEdgeRatio: input ratio
        """
        self.recEdgeRatio = recEdgeRatio
        assert self.recEdgeRatio >= 1, 'Ratio must be larger or equal 1'
        self.setRatio()

    def setRatio(self):
        """
        set the threshold ratio between width and height for detecting rectange
            :param self: 
        """ 
        self.threshold_1 = float(self.recEdgeRatio)
        self.threshold_2 = float(1/self.recEdgeRatio)
        
    def detect(self, cnt):
        """
        check whether contour is rectangle with definded ratio between width and 
        height, if true return contour centroid, size, rectangle's corners
            :param cnt: input cotour, found by cv2.findContour()
        return:
            :detected: list results contain 4 elements:
                0 - Boolean, indicate contour is rectangle or not
                1 - centroid position
                2 - contour size
                3 - rectangle corners
        """ 
        detected = [False, None, None, None]
        rect = cv2.minAreaRect(cnt)
        w, h = rect[1]
        ar = w / float(h)
        if ar >= self.threshold_1 or ar < self.threshold_2:
                detected[0] = True
                detected[1] = rect[0]
                detected[2] = w*h
                box = cv2.boxPoints(rect)
                detected[3] = np.int0(box)
        return detected