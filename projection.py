import cv2
import numpy as np

# Note: Once we have defined the transformation and restricted zone,
# should use bounding box of people in original frame
# maybe take the average of the bottom line to get a center point where the feet are located
# then transform that point to new perspective and determine if it's contained in the restricted zone

# looks like there's a bit of camera drift as the video continues as the zone line shifts slightly

# May want a separate calibration/setup procedure to select source/destination points and zone boundary

# configurations
frame_rate = 60.0  # frames per second
video_name = 'worker-zone-detection.mp4'  # taken from https://github.com/intel-iot-devkit/sample-videos
boundingbox_shift = -50  # number of pixels to y coordinate of centroid to match with foot location


# define restricted zone - this should be defined in transformed perspective coordinate system
class RestrictedZone:
    def __init__(self, pts):
        self.pts = pts  # points for border of polygon - should be in order of how to connect
        self.color = (0, 0, 255)  # BGR
        self.alpha = 0.4  # Transparency factor
        self.warning_dist = 15  # pixel distance from edge to create a warning zone

    # overlay the zone on the image with some transparency
    def overlay_image(self, img):
        overlay = img.copy()
        cv2.fillPoly(overlay, [self.pts], self.color)
        image_overlaid = cv2.addWeighted(overlay, self.alpha, img, 1 - self.alpha, 0)
        return image_overlaid

    # determine if the point is within the zone
    # Return True if in the zone, False if not in zone, otherwise return string 'Warning Zone'
    def is_point_in_zone(self, test_pt):
        dist = cv2.pointPolygonTest(self.pts, test_pt, True)
        # if True parameter, it will
        # return the distance from the edge, could be useful for defining a buffer zone at the edges
        if dist >= 0:  # positive or at border 0 means inside the polygon
            return True
        else:
            if abs(dist) <= self.warning_dist:  # if the centroid distance from edge is within warning distance
                return "Warning Zone"
            else:
                return False


# Defining a projection class
class Projection:
    def __init__(self, size, src, dst):
        self.size = size  # (width, height) to resize image to
        self.src = src
        self.dst = dst
        self.matrix, status = cv2.findHomography(src, dst)

    # apply transformation to input image
    def transform_frame(self, input_img):
        transformed = cv2.warpPerspective(input_img, self.matrix, self.size)
        return transformed

    # apply transformation to centroid
    def transform_pt(self, pt):
        pt3 = [[pt[0]], [pt[1]], [1]]  # add in the 3rd coordinate
        transformed_pt = np.matmul(self.matrix, pt3)  # matrix multiply
        transformed_pt = np.divide(transformed_pt, transformed_pt[2][0])  # normalize third coordinate to 1
        # print(transformed_pt)
        return (int(transformed_pt[0][0]), int(transformed_pt[1][0]))  # return just x,y


# Main execution code

# source/destination coordinates for the worker video demonstration
# format is (x,y) locations
# source coordinates - box
source = np.float32([112, 211,  # top left
                     332, 401,  # bottom left
                     755, 241,  # bottom right
                     461, 145, ]).reshape((4, 2))  # top right
# destination coordinates - box
dest = np.float32([200, 200,  # top left
                   200, 400,  # bottom left
                   400, 400,  # bottom right
                   400, 200, ]).reshape((4, 2))  # top right

# restricted zone points
zone_points = np.array([[156, 433], [304, 351],
                        [304, 0], [191, 0],
                        [118, 403]],
                       np.int32).reshape((-1, 1, 2))

red_zone = RestrictedZone(zone_points)
my_transform = Projection((960, 540), source, dest)
print(my_transform.matrix)

# set up output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('overlay_output.mp4', fourcc, frame_rate, my_transform.size)
out_tf = cv2.VideoWriter('transformed_output.mp4', fourcc, frame_rate, my_transform.size)

# person detector - fill in with backpack later
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# load in the video and read frame by frame
vidcap = cv2.VideoCapture(video_name)
success, image = vidcap.read()  # read the first frame
counter = 0  # count frames, 0-indexed
while success:  # loop through all the frames
    if True:  # counter % 20 == 0: #adjust if it's too slow
        if counter % 20 == 0:
            print(counter)
        # resize the image
        img_resized = cv2.resize(image, my_transform.size)

        # transform with the matrix
        tf_img = my_transform.transform_frame(img_resized)

        # create new image with zone overlay
        image_new = red_zone.overlay_image(tf_img)

        # detect people in the image
        # returns the bounding boxes for the detected objects in the original coordinate system
        boxes, weights = hog.detectMultiScale(img_resized, winStride=(8, 8))
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        for (xA, yA, xB, yB) in boxes:
            # display the detected boxes in the colour picture
            # cv2.rectangle(img_resized, (xA, yA), (xB, yB),
            #               (0, 255, 0), 2)
            centroid = (int((xA + xB) / 2), int(yB) + boundingbox_shift)  # centroid in original
            transformed_centroid = my_transform.transform_pt(centroid)  # centroid in new coordinate system
            # print(transformed_centroid)
            zone_flag = red_zone.is_point_in_zone(transformed_centroid)
            cv2.drawMarker(img_resized, centroid, (255,0,0),thickness=2)
            cv2.drawMarker(image_new, transformed_centroid, (255, 0, 0), thickness=2)

            # add some text next to centroid whether it's in the zone or not
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            thickness = 2
            if zone_flag == "Warning Zone":
                color = (255, 255, 0)
            elif zone_flag:
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)
            cv2.putText(image_new, str(zone_flag), (transformed_centroid[0] + 2, transformed_centroid[1] + 2), font,
                        fontScale, color, thickness, cv2.LINE_AA)
            cv2.putText(img_resized, str(zone_flag), (centroid[0] + 2, centroid[1] + 2), font,
                        fontScale, color, thickness, cv2.LINE_AA)

        # write the output frame to video
        out_tf.write(image_new)
        out.write(img_resized)

        cv2.imshow('transformed', image_new)  # Transformed Capture
        cv2.imshow('original', img_resized)  # Transformed Capture
        cv2.waitKey(1)

    # read the next frame
    success, image = vidcap.read()
    counter = counter + 1

# end loop, release objects
out.release()
vidcap.release()
cv2.destroyAllWindows()
