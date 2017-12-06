import argparse
from datetime import datetime
import numpy as np
import cv2
import glob
from moviepy.editor import VideoFileClip


# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([[0, 0, 0]], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None


class CameraCalibration:
    def __init__(self, img_dim, chessboard_dim, glob_imgs_path):
        self.img_height = img_dim[0]
        self.img_width = img_dim[1]
        self.chessboard_rows = chessboard_dim[0]
        self.chessboard_cols = chessboard_dim[1]
        self.glob_imgs_path = glob_imgs_path

    def find_objpoints_imgpoints(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.chessboard_rows * self.chessboard_cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_cols, 0:self.chessboard_rows].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob(self.glob_imgs_path)

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.chessboard_cols, self.chessboard_rows), None)

            # assert ret is True, "Finding chessboard corners failed. Please check your arguments."
            # If found, add object points, image points
            if ret is True:
                objpoints.append(objp)
                imgpoints.append(corners)

        return objpoints, imgpoints

    # Do camera calibration given object points and image points and return camera matrix and distortion coeff.
    def calibrate_camera(self, objpoints, imgpoints):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (self.img_height, self.img_width),
                                                           None, None)
        return mtx, dist

    @staticmethod
    def undistort_img(img, camera_mtx, dist_coeff):
        return cv2.undistort(img, camera_mtx, dist_coeff, None, camera_mtx)

    def calibrate(self):
        objpoints, imgpoints = self.find_objpoints_imgpoints()

        camera_mtx, dist_coeff = self.calibrate_camera(objpoints, imgpoints)

        return camera_mtx, dist_coeff


class Thresholding:
    def __init__(self, img):
        """img has to be RGB image."""
        self.img = img

    def abs_sobel_thresh(self, orient='x', sobel_kernel=3, thresh=(30, 100)):
        # Calculate directional gradient
        # 1) Convert to grayscale
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'x':
            sobel = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        else:
            sobel = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # 3) Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel)
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        # 5) Create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max
        sobel_binary = np.zeros_like(scaled_sobel)
        sobel_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        return sobel_binary

    def mag_thresh(self, sobel_kernel=3, mag_thresh=(30, 100)):
        # Calculate gradient magnitude
        # Convert to grayscale
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        return binary_output

    def dir_threshold(self, sobel_kernel=3, thresh=(0.7, 1.3)):
        # Calculate gradient direction
        # Grayscale
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        # Return the binary image
        return binary_output

    def apply_gradient_mag_and_dir_threshold(self, kernel_size=9):
        # Apply each of the thresholding functions
        gradx = self.abs_sobel_thresh(orient='x', sobel_kernel=kernel_size)
        grady = self.abs_sobel_thresh(orient='y', sobel_kernel=kernel_size)
        mag_binary = self.mag_thresh(sobel_kernel=21)
        dir_binary = self.dir_threshold(sobel_kernel=15)

        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1) & (gradx == 1))] = 1

        return combined

    def apply_thresholding(self, s_thresh=(150, 255), h_thresh=(15, 100)):

        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(self.img, cv2.COLOR_RGB2HLS).astype(np.float)
        h_channel = hls[:, :, 0]
        s_channel = hls[:, :, 2]

        # Sobel grad mag dir
        grad_mag_dir_binary = self.apply_gradient_mag_and_dir_threshold()

        # Threshold color channel, s_channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        # Threshold color channel, h_channel
        h_binary = np.zeros_like(h_channel)
        h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1

        binary = np.zeros_like(s_binary)
        binary[(grad_mag_dir_binary == 1) | ((s_binary == 1) & (h_binary == 1))] = 1

        return binary


class PerspectiveTransform:
    def __init__(self, src_pts, dst_pts):
        self.src_pts = src_pts
        self.dst_pts = dst_pts

    def get_perspective_transform_mtx(self):
        M = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        M_inv = cv2.getPerspectiveTransform(self.dst_pts, self.src_pts)

        return M, M_inv

    @staticmethod
    def transform_perspective(img, M):
        # Assuming cv2.imread() has been used
        img_size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        return warped


class AdvancedLaneDetection:
    def __init__(self, video_path, camera_mtx, dist_coeff):
        self.video_path = video_path

        self.clip = VideoFileClip(self.video_path) #  .subclip(40, 45)
        self.img_height = self.clip.size[1]
        self.img_width = self.clip.size[0]
        self.ploty = np.linspace(0, self.img_height - 1, self.img_height)

        self.camera_mtx = camera_mtx
        self.dist_coeff = dist_coeff
        #
        # self.src_pts = np.float32(
        #     [[710, 460],
        #      [1150, self.img_height],
        #      [195, self.img_height],
        #      [580, 460]
        #      ])
        self.src_pts = np.float32(
            [[(self.img_width / 2) - 60, self.img_height / 2 + 100],
             [((self.img_width / 6) - 18), self.img_height],
             [(self.img_width * 5 / 6) + 80, self.img_height],
             [(self.img_width / 2 + 70), self.img_height / 2 + 100]
             ])

        self.dst_pts = np.float32(
            [[(self.img_width / 4), 0],
             [(self.img_width / 4), self.img_height],
             [(self.img_width * 3 / 4), self.img_height],
             [(self.img_width * 3 / 4), 0]
             ])
        self.nwindows = 12
        # Set height of windows
        self.window_height = np.int(self.img_height / self.nwindows)
        self.margin = 100
        self.minpix = 40

        self.frame_avg = 25  # Averaging interval is 1 second
        self.frame_counter = 0

        self.left_line = Line()
        self.right_line = Line()
        self.perspective_trans = PerspectiveTransform(self.src_pts, self.dst_pts)
        self.M, self.M_inv = self.perspective_trans.get_perspective_transform_mtx()

        self.perform_blind_search = True
        self.leftx_current = 0
        self.rightx_current = 0

        self.vertices = np.array([[[(self.img_width / 2 + 75), self.img_height / 2 + 120],
                                   [(self.img_width * 5 / 6) + 200, self.img_height],
                                   [((self.img_width / 6) - 30), self.img_height],
                                   [(self.img_width / 2) - 75, self.img_height / 2 + 120]]],
                                 dtype=np.int32)

    def process_video(self, output_path):
        output_clip = self.clip.fl_image(self.process_frame)  # NOTE: this function expects color images!!
        output_clip.write_videofile(output_path, audio=False)

    def blind_search(self, binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        return leftx_base, rightx_base

    def cal_curvature(self):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        left_fitx = self.left_line.bestx
        right_fitx = self.right_line.bestx
        y_eval = int(max(self.ploty))

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(self.ploty * ym_per_pix, left_fitx * xm_per_pix, 2)

        right_fit_cr = np.polyfit(self.ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (
            2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = (
                             (1 + (
                                 2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[
                                     1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        # Now our radius of curvature is in meters
        # print(left_curverad, 'm', right_curverad, 'm')

        curvature = left_curverad if left_curverad <= right_curverad else right_curverad
        curvature = round(curvature)
        curvature_text = "Lane's Radius of curvature: {}m ".format(curvature)

        # print(curvature_text)
        dist_between_two_lane_lines = int(right_fitx[y_eval] - left_fitx[y_eval]) * xm_per_pix

        offset_from_lane_center = int(self.img_width / 2) - int(
            (right_fitx[y_eval] - left_fitx[y_eval]) / 2 + left_fitx[y_eval])
        offset_from_lane_center_in_meters = offset_from_lane_center * xm_per_pix

        if offset_from_lane_center_in_meters < 0:
            dist_from_center_text = "Vehicle is {:.2}m left of lane center, dist between lines {:.2}".format(
                abs(offset_from_lane_center_in_meters), dist_between_two_lane_lines)
        else:
            dist_from_center_text = "Vehicle is {:.2}m right of lane center, dist between lines {:.2}".format(
                abs(offset_from_lane_center_in_meters), dist_between_two_lane_lines)

        # print(dist_from_center_text)

        return curvature_text, dist_from_center_text, abs(offset_from_lane_center_in_meters)

    def region_of_interest(self, img):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        # defining a blank mask to start with
        mask = np.zeros_like(img)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, self.vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def process_frame(self, frame):

        img_undistorted = CameraCalibration.undistort_img(frame, self.camera_mtx, self.dist_coeff)

        img_binary_threshold = Thresholding(img_undistorted).apply_thresholding()
        # img_binary_threshold = self.region_of_interest(img_binary_threshold)
        # cv2.imwrite("output" + str(self.frame_counter) + ".jpg", img_binary_threshold)

        binary_warped = self.perspective_trans.transform_perspective(img_binary_threshold, self.M)

        # Perform blind search only after frame_avg
        if self.frame_counter % self.frame_avg == 0:
            leftx_base, rightx_base = self.blind_search(binary_warped)
        else:
            leftx_base = int(self.left_line.bestx[-1])
            rightx_base = int(self.right_line.bestx[-1])

        # Current positions to be updated for each window
        self.leftx_current = leftx_base
        self.rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()

        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * self.window_height
            win_y_high = binary_warped.shape[0] - window * self.window_height
            win_xleft_low = self.leftx_current - self.margin
            win_xleft_high = self.leftx_current + self.margin
            win_xright_low = self.rightx_current - self.margin
            win_xright_high = self.rightx_current + self.margin
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &
                              (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &
                               (nonzerox < win_xright_high)).nonzero()[0]

            # print(good_left_inds.shape, good_left_inds)

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                self.leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                self.rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        self.left_line.allx, rej_ids = outlier_removal(nonzerox[left_lane_inds])
        self.left_line.ally = nonzeroy[left_lane_inds]
        self.left_line.ally = np.delete(self.left_line.ally, rej_ids)

        self.right_line.allx, rej_ids = outlier_removal(nonzerox[right_lane_inds])
        self.right_line.ally = nonzeroy[right_lane_inds]
        self.right_line.ally = np.delete(self.right_line.ally, rej_ids)

        # Fit a second order polynomial to each
        if len(self.left_line.allx) > 0 and len(self.left_line.ally) > 0:
            left_fit = np.polyfit(self.left_line.ally, self.left_line.allx, 2)
            self.left_line.current_fit.append(np.array([left_fit[0], left_fit[1], left_fit[2]]))
            self.left_line.best_fit = np.mean(self.left_line.current_fit, axis=0)
            if len(self.left_line.current_fit) > 1:
                self.left_line.diffs = np.append(self.left_line.diffs,
                                                 [self.left_line.current_fit[-2] - self.left_line.current_fit[-1]])

            left_fitx = left_fit[0] * self.ploty ** 2 + left_fit[1] * self.ploty + left_fit[2]
        else:
            # use the last fit if we don't have any data to fit in this frame
            # left_fitx = self.left_lane.recent_xfitted[len(self.left_lane.recent_xfitted) - 1]
            left_fit = self.left_line.best_fit
            left_fitx = left_fit[0] * self.ploty ** 2 + left_fit[1] * self.ploty + left_fit[2]

        if len(self.right_line.allx) > 0 and len(self.right_line.ally) > 0:
            right_fit = np.polyfit(self.right_line.ally, self.right_line.allx, 2)
            self.right_line.current_fit.append(np.array([right_fit[0], right_fit[1], right_fit[2]]))
            self.right_line.best_fit = np.mean(self.right_line.current_fit, axis=0)
            if len(self.right_line.current_fit) > 1:
                self.right_line.diffs = np.append(self.right_line.diffs,
                                                  [self.right_line.current_fit[-2] - self.right_line.current_fit[-1]])
            right_fitx = right_fit[0] * self.ploty ** 2 + right_fit[1] * self.ploty + right_fit[2]
        else:
            # use the last fit if we don't have any data to fit in this frame
            # right_fitx = self.right_lane.recent_xfitted[len(self.right_lane.recent_xfitted) - 1]
            right_fit = self.right_line.best_fit
            right_fitx = right_fit[0] * self.ploty ** 2 + right_fit[1] * self.ploty + right_fit[2]

        if self.frame_counter % self.frame_avg == 0:
            self.right_line.recent_xfitted.clear()
            self.right_line.recent_xfitted.clear()

        self.left_line.recent_xfitted.append(left_fitx)
        entries_count = len(self.left_line.recent_xfitted)
        if entries_count > 1:
            w = [.5 / (entries_count - 1) * i for i in range(0, entries_count - 1)]
            w.append(1)
            self.left_line.bestx = np.average(self.left_line.recent_xfitted, axis=0,
                                              weights=w)
        else:
            self.left_line.bestx = left_fitx

        self.right_line.recent_xfitted.append(right_fitx)
        entries_count = len(self.right_line.recent_xfitted)
        if entries_count > 1:
            w = [.5 / (entries_count - 1) * i for i in range(entries_count - 1)]
            w.append(1)
            self.right_line.bestx = np.average(self.right_line.recent_xfitted, axis=0,
                                               weights=w)
        else:
            self.right_line.bestx = right_fitx

        new_img = self.unwarp_binary_map_on_img(binary_warped, img_undistorted)

        curvature_text, dist_from_center_text, dist_from_center = self.cal_curvature()

        cv2.putText(new_img, curvature_text, (30, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv2.putText(new_img, dist_from_center_text, (30, 130), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        self.frame_counter += 1

        return new_img

    def unwarp_binary_map_on_img(self, binary_warped, img_undistorted):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        red_blue_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        left_fitx = self.left_line.recent_xfitted[-1]
        right_fitx = self.right_line.recent_xfitted[-1]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        red_blue_warp[self.left_line.ally, self.left_line.allx] = [200, 0, 0]
        red_blue_warp[self.right_line.ally, self.right_line.allx] = [0, 0, 200]

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 150, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.M_inv, (self.img_width, self.img_height))
        # Combine the result with the original image
        result = cv2.addWeighted(img_undistorted, 1, newwarp, 0.3, 0)

        blue_inv_warp = cv2.warpPerspective(red_blue_warp, self.M_inv, (self.img_width, self.img_height))
        # Combine the result with the original image
        result = cv2.addWeighted(result, .8, blue_inv_warp, 1, 0)

        return result


def outlier_removal(a):
    q75, q25 = np.percentile(a, [75, 25])
    iqr = q75 - q25

    _min = q25 - (iqr * 1.3)
    _max = q75 + (iqr * 1.3)

    r = []
    rej_ids = []

    for i in range(len(a)):
        if _min <= a[i] <= _max:
            r.append(a[i])
        else:
            rej_ids.append(i)

    return r, rej_ids


def main():
    """Parses the command line options and kick-start the event loop to send data to BioTracker Server simulating one
    or more clients."""

    parser = argparse.ArgumentParser()

    parser.add_argument("-hgt", "--imgHeight", help="The height of the images, default=720.",
                        type=int, default=720)

    parser.add_argument("-wd", "--imgWidth", help="The width of the images, default=1280.",
                        type=int, default=1280)

    parser.add_argument("-r", "--chessboardRows", help="The rows of the chessboard calibration images, default=6.",
                        type=int, default=6)

    parser.add_argument("-c", "--chessboardCols", help="The cols of the chessboard calibration images, default=9.",
                        type=int, default=9)

    parser.add_argument("-cp", "--calibrationPath", help="The height of the images, default=720.",
                        type=str, default='')

    parser.add_argument("-in", "--inputVideoPath", help="The path to the input video to be processed.",
                        type=str, default='')

    parser.add_argument("-out", "--outputVideoPath", help="The path to the where to store output video.",
                        type=str, default='')

    args = parser.parse_args()

    print(args)

    assert args.calibrationPath != '', "The path to calibration images can't be empty"
    assert args.inputVideoPath != '', "The path to input video can't be empty"
    assert args.outputVideoPath != '', "The path to output video can't be empty"

    camera_mtx, dist_coeff = CameraCalibration((args.imgHeight, args.imgWidth),
                                               (args.chessboardRows, args.chessboardCols),
                                               args.calibrationPath).calibrate()
    print("Camera Mtx", camera_mtx)
    print("Distortion Coefficient", dist_coeff)
    # img = cv2.imread('test_images/test5.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    AdvancedLaneDetection(args.inputVideoPath, camera_mtx, dist_coeff).process_video(args.outputVideoPath)

    # cv2.imwrite("output.jpg", result)


if __name__ == '__main__':
    main()
