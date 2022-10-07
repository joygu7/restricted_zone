import numpy as np
import sys
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from projection import Projection

# Grab one frame from the camera feed
# Grab image of floor plan/ or leave as blank grid

# Top as camera, bottom as floor
# toggle between selecting control points and restricted zone area ()

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton
from PyQt5.QtGui import QPixmap

class App(QWidget):
    def __init__(self, videoname):
        super().__init__()

        # camera
        self.pointcounter = 0
        vidcap = cv2.VideoCapture(videoname)
        success, frame = vidcap.read() #read first frame
        self.framesize = (960, 540)
        self.frame_orig = cv2.resize(frame, self.framesize) #resize
        self.frame = cv2.resize(frame, self.framesize) #resize
        self.pix = self.convert_frame_to_pix(self.frame)
        self.camera = QLabel(self)
        self.camera.setPixmap(self.pix)
        self.camera.resize(self.framesize[0], self.framesize[1])
        self.camera.move(20,90)
        self.camera.mousePressEvent = self.getCameraPts
        self.camerapts = []

        # floor plan
        self.pointcounter2 = 0
        frame2 = cv2.imread('grid-texture.jpg')
        self.frame2_orig = cv2.resize(frame2, self.framesize)  # resize
        self.frame2 = cv2.resize(frame2, self.framesize)  # resize
        self.pix2 = self.convert_frame_to_pix(self.frame2)
        self.floor = QLabel(self)
        self.floor.setPixmap(self.pix2)
        self.floor.resize(self.framesize[0], self.framesize[1])
        self.floor.move(20, 90)
        self.floor.mousePressEvent = self.getFloorPts
        self.floorpts = []

        #zone definition - need to do transformation first?
        self.frame3 = None
        self.pix3 = None
        self.projection = None
        self.zone = QLabel(self)
        self.zone.resize(self.framesize[0], self.framesize[1])
        self.zone.move(20, 90)
        self.zone.mousePressEvent = self.getZonePts

        #buttons
        self.camera_button = QPushButton(self)
        self.camera_button.setText("Camera Pt Collection")
        self.camera_button.move(64, 32)
        self.camera_button.clicked.connect(self.camera_button_click)

        self.floor_button = QPushButton(self)
        self.floor_button.setText("Floor Pt Collection")
        self.floor_button.move(4*64, 32)
        self.floor_button.clicked.connect(self.floor_button_click)

        self.zone_button = QPushButton(self)
        self.zone_button.setText("Zone Pt Collection")
        self.zone_button.move(8 * 64, 32)
        self.zone_button.clicked.connect(self.zone_button_click)

        self.clear_button = QPushButton(self)
        self.clear_button.setText("Clear")
        self.clear_button.move(10 * 64, 32)
        self.clear_button.clicked.connect(self.clear_button_click)

        self.floor.setHidden(True)
        self.camera.setHidden(False)
        self.zone.setHidden(True)
        self.show()

    def getCameraPts(self, event):
        x = event.pos().x()
        y = event.pos().y()
        self.pointcounter = self.pointcounter + 1
        self.camerapts.append([x,y])
        print(self.camerapts)

        coord_str = str(self.pointcounter)+':(' + str(x) + ',' + str(y) + ')'
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        thickness = 2
        color = (0, 255, 0)
        self.frame = cv2.putText(self.frame, coord_str, (x-10,y-15), font,
                        fontScale, color, thickness, cv2.LINE_AA )
        self.frame = cv2.drawMarker(self.frame, (x, y),
                                    color, markerType=cv2.MARKER_DIAMOND, thickness=thickness)
        self.pix = self.convert_frame_to_pix(self.frame)
        self.camera.setPixmap(self.pix)

        print("(x,y)= (%d, %d)" %(x,y))


    def getFloorPts(self, event):
        x = event.pos().x()
        y = event.pos().y()
        self.pointcounter2 = self.pointcounter2 + 1
        self.floorpts.append([x, y])
        print(self.floorpts)

        coord_str = str(self.pointcounter2)+':(' + str(x) + ',' + str(y) + ')'
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        thickness = 2
        color = (0, 255, 0)
        self.frame2 = cv2.putText(self.frame2, coord_str, (x-10,y-15), font,
                        fontScale, color, thickness, cv2.LINE_AA )
        self.frame2 = cv2.drawMarker(self.frame2, (x, y),
                                    color, markerType=cv2.MARKER_DIAMOND, thickness=thickness)
        self.pix2 = self.convert_frame_to_pix(self.frame2)
        self.floor.setPixmap(self.pix2)

        print("(x,y)= (%d, %d)" %(x,y))

    def getZonePts(self, event):
        print(0)

    def convert_frame_to_pix(self, frame):
        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        img = QtGui.QImage(frame, width, height, bytesPerLine, QtGui.QImage.Format_RGB888).rgbSwapped()
        pix = QtGui.QPixmap.fromImage(img)
        return pix

    def camera_button_click(self):
        print('click1')
        self.floor.setHidden(True)
        self.camera.setHidden(False)
        self.zone.setHidden(True)

    def floor_button_click(self):
        print('click2')
        self.floor.setHidden(False)
        self.camera.setHidden(True)
        self.zone.setHidden(True)

    def zone_button_click(self):
        print('click3')
        print(self.camerapts)
        src = np.float32(self.camerapts)
        dst = np.float32(self.floorpts)
        self.projection = Projection(self.framesize, src, dst)
        self.frame3 = self.projection.transform_frame(self.frame_orig)
        self.pix3 = self.convert_frame_to_pix(self.frame3)
        self.zone.setPixmap(self.pix3)
        self.floor.setHidden(True)
        self.camera.setHidden(True)
        self.zone.setHidden(False)

    def clear_button_click(self):
        print('clear')
        self.frame = np.copy(self.frame_orig)
        self.pix = self.convert_frame_to_pix(self.frame)
        self.camera.setPixmap(self.pix)
        self.camerapts = []
        self.pointcounter = 0

        self.frame2 = np.copy(self.frame2_orig)
        self.pix2 = self.convert_frame_to_pix(self.frame2)
        self.floor.setPixmap(self.pix2)
        self.floorpts = []
        self.pointcounter2 = 0


        self.floor.setHidden(True)
        self.camera.setHidden(False)
        self.zone.setHidden(True)



if __name__ == '__main__':
    videoname = 'worker-zone-detection.mp4'
    app = QApplication(sys.argv)
    ex = App(videoname)
    sys.exit(app.exec_())