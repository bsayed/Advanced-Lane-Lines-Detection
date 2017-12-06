

# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.png "Undistorted"
[road_undistorted]: ./output_images/road_undistorted.png ""
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/binary_combo_example.png "Binary Example"
[warped]: ./output_images/warped.png "Warp Example"
[warped2]: ./output_images/warped2.png "Warp Example"
[image5]: ./output_images/color_fit_lines.png "Fit Visual"
[image6]: ./output_images/example_output.png "Output"
[video1]: ./project_video_output.mp4 "Video"




### Camera Calibration

#### 1. How the camera matrix and distortion coefficients were computed. An example of a distortion corrected calibration image is provided.

The code for this step is contained in the CameraCalibration class in lines 43 through 85 of the file called `advanced_lane_finiding.py`). The constructor of the class takes the image dimension as a tuple (height,width), the chessboard dimension as a tuple (row,col), and the last argument is the path to the where the calibration images located. The last argument should be in filename globbing format.
The class provide the `calibrate()` fucntion that perfrom the actual calibration of the camera used to take the chessboard photos.
The `calibrate()` function starts by calling `find_objpoints_imgpoints()` fucntion. The `find_objpoints_imgpoints()` function starts by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

The `find_objpoints_imgpoints()` function returns the detected `objpoints` and `imgpoints`.
I then used the `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients by calling the `calibrate_camera()` passing the `objpoints` and `imgpoints` as arguments. Internally the `calibrate_camera` function uses the `cv2.calibrateCamera()` function to perform the actual calibration. The `calibrate_camera()` function returns the `camera_mtx` and the distortion coefficient `dist_coeff`. The CameraCalibration class provides a static method `undistort_img` that takes an image plus the camera matrix and the distortion coefficients to undsitort the image. I applied this function on the following image and obtained this result: 

![alt text][image1]

### Pipeline (single image)

### Overview of the pipeline
The `AdvancedLaneDetection` class defined at lines 203 through 482 in the file 
`advanced_lane_finding.py`, provides the main functionality for processing images to 
detect lane lines. This class uses other helper classes, also defined in the same file,
namely `Thresholding` class, `Line` class, and the `PerspectiveTransform` class.
The `Line` class represent a detected line either left line or right line of the lane.
The `Thresholding` class provides the functionality to apply color transforms,
gradients and other methods to create a thresholded binary image. 
Finally, the `PerspectiveTransform` class provides the functionality to get the
perspective transform matrix and its inverse, plus it provides a static method to
perform the actual perspective transform on an image.
  

#### 1. Provide an example of a distortion-corrected image.
I used the `undistort_img()` function defined as a static method in the 
`CameraCalibration` class. This method takes an image as an argument plus
the camera matrix and the distortion coefficients to un-distort the image.
The following is an example output of that function:

![alt text][road_undistorted]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image
(thresholding steps at lines 88 through 181 in `advanced_lane_finding.py`).
I combined the absolute sobel thresholding in both the x and y direction with 
magnitude and direction thresholding, and the color thresholding in both saturation 
and hue channels after converting the image to HLS color space. The main focus was to
make sure the lane lines are detected at but with less noise from the shadows as much as
possible.
Here's an example of my output for this step that contains shows of the tree.  


![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is presented in the `PerspectiveTransform` class.
The class has two functions, the first is `get_perspective_transform_mtx()` which calculates
the perspective transform matrix and its inverse. The second, is `transform_perspective()`
which is a static method that perform the actual perpective transform given an image and
a perspective transform matrix. The class appears in lines 184 through 200
in the file `advanced_lane_finding.py`. 
I chose the hardcode the source and destination points in the following manner:

```python
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
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 320, 0        | 
| 195, 720      | 320, 720      |
| 1146, 720     | 960, 720      |
| 710, 460      | 960, 0        |

The following 2 images gives an example of the perspective transform:

![alt text][warped]

![alt text][warped2]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
I start by using the histogram calculation method to blindly identify the beginning of 
lane-line pixels in the fist frame, then I use that to guide the search for the lane-lines pixels,
then I fit the found pixels with 2nd degree polynomial. I used the `Line` class, found in the
`advanced_lane_finding.py` file at lines 10 through 31 to keep information about the found
pixels and the fitted pixels over several frames as a kind of optimization. The last values
of the polynomial fit are used to guide the search in the subsequent frames, however, the blind
search restart every 25 frames (one second in the video time). This is presented in lines 352 to 361.
The actual fitting happens between line 409 and 425 and the search for the pixels happens between
line 360 and 406, all of lines are part of the `process_frame()` function of the 
`AdvancedLaneDetection` class.
The following image outlines the search for line pixels process. (note that this image is from
the harder_chanllenge video)

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
The calculation of the radius of curvature is presented in lines 273 through 317 in the 
`cal_curvature()` function that belongs to the `AdvancedLaneDetection` class.
This method returns the text for both the radius of curvature and the vehicle position with
respect to the lane center.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 448 through 477 in my code in `advanced_lane_finding.py` 
in the function `unwarp_binary_map_on_img()`.  
Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Although initially I have taken the convolution approach as it did work better for the
challenge videos, for some reason it did not perform well on the main project video. I ended
up reducing the size of the convolution window and quickly found my self doing something 
similar to the regular search boxes method. So I have decided to switch back. 
One of the techniques that I have tried to reduce the effect of the background noise on finding
the lane pixels is "region-of-interest". It is already implemented in the `advanced_lane_finding.py` 
as part of the `AdvancedLaneDetection` class. It minimized the effect of the trees in the harder
challenge video but wasn't enough to get a reasonable performance by itself, specially in some terns, the
lane lines are very hard to be seen, even for humans. 
Also extremely dark frames like in the challenge video, when the vehicle is crossing under a
bridge, the lane detection couldn't find any pixels, the challenge here is to find the right
trade off between reducing the effect of shadows but at the same time be able to find lane lines
in dark frames.

One of the techniques that worked really well, is the averaging of the fitting data over 
certain number of frames e.g. 25 frames, this reduced the effect of jitter and made the 
lane marking looks smooth.

Extra stuff that I did later:

 1. Implemented a outlier removal function (file "advanced_lane_finding.py" lines 511 through 527)
 2. Implemented decaying weighted average to average the previous fittings giving more recent good fit a higher weight and older fits less weight (file "advanced_lane_finding.py" lines 451 to 453 and lines 461 to 463). If we couldn't find any pixels in the current frame we use the best_fit that we currently have to apply the polynomial fitting (lines 427 to 428 and 441 to 442), the best_fit is the mean of all the previous good fittings.
 3. Calculated the distance between the two lane lines with the intend to use it as a sanity check but did not need to use it.
