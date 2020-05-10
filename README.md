# Self-driving-car-Advanced


**The goals / steps of this project are the following:**
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.You only need to compute these once
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view"). Pick four points in a trapezoidal shape (similar to region masking)
	1. Plotting a histogram of where the binary activations occur across the image
	2. then use sliding windows moving upward in the image (further along the road) to determine where the lane lines go
	3. Fit a polynomial
* Determine the [curvature](https://www.intmath.com/applications-differentiation/8-radius-curvature.php) of the lane and vehicle position with respect to center. Keep in mind
	1. the lane is about 30 meters long and 3.7 meters wide
	2. minimum lane width of 12 feet or 3.7 meters, and the dashed lane lines are 10 feet or 3 meters long each
	3. our camera image has 720 relevant pixels in the y-dimension (remember, our image is perspective-transformed!)
	4. and we'll say roughly 700 relevant pixels in the x-dimension (the example of fake generated data used from 200 pixels on the left to 900 on the right, or 700).

# Perspective Transform

The code for my perspective transform includes a function called `perspective_transform()`, which appears in lines 131 through 134 in the file `Proyect2.py`.  The `warper()` function takes as inputs an image (`img`), as well as the perspective transform matrix (`M`).  I chose the hardcode the source and destination points in the following manner:

```python
transform_src = np.float32([
	[180,720],[575,460],[710,460],[1133,720]
	])
	
transform_dst = np.float32([
	[300,720],[290,50],[980,50],[950,720]
	])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 180, 720      | 300, 720      | 
| 575, 460      | 290, 50      |
| 710, 460     | 980, 50      |
| 1133, 720    | 950, 720        |

### Perspective Transform

| Original |Perspective transform|
------------ | -------------
<img src="./output_images/perspective_transform/PerspectiveTransformModel.jpg" width="400" height1="100">|<img src="./output_images/perspective_transform/PerspectiveTransformModelOutput.jpg" width="400" height1="100">

# Color Transform
| Perspective transform | S channel Only |Sobel + thresholds | Combined | Yellow & White mask only|
------------ | ------------- | ------------- | ------------- | -------------
<img src="./output_images/perspective_transform/straight_lines1.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/straight_lines1_SchannelOnly.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/straight_lines1_FullColorTransf.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/straight_lines1_Combined.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/straight_lines1_Option2.jpg" width="150" height1="100">|
<img src="./output_images/perspective_transform/straight_lines2.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/straight_lines2_SchannelOnly.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/straight_lines2_FullColorTransf.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/straight_lines2_Combined.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/straight_lines2_Option2.jpg" width="150" height1="100">|
<img src="./output_images/perspective_transform/test1.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test1_SchannelOnly.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test1_FullColorTransf.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test1_Combined.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test1_Option2.jpg" width="150" height1="100">|
<img src="./output_images/perspective_transform/test2.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test2_SchannelOnly.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test2_FullColorTransf.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test2_Combined.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test2_Option2.jpg" width="150" height1="100">|
<img src="./output_images/perspective_transform/test3.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test3_SchannelOnly.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test3_FullColorTransf.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test3_Combined.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test3_Option2.jpg" width="150" height1="100">|
<img src="./output_images/perspective_transform/test4.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test4_SchannelOnly.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test4_FullColorTransf.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test4_Combined.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test4_Option2.jpg" width="150" height1="100">|
<img src="./output_images/perspective_transform/test5.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test5_SchannelOnly.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test5_FullColorTransf.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test5_Combined.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test5_Option2.jpg" width="150" height1="100">|
<img src="./output_images/perspective_transform/test6.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test6_SchannelOnly.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test6_FullColorTransf.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test6_Combined.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test6_Option2.jpg" width="150" height1="100">|
<img src="./output_images/perspective_transform/test_challenge0.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test_challenge0_SchannelOnly.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test_challenge0_FullColorTransf.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test_challenge0_Combined.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test_challenge0_Option2.jpg" width="150" height1="100">|
<img src="./output_images/perspective_transform/test_challenge1.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test_challenge1_SchannelOnly.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test_challenge1_FullColorTransf.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test_challenge1_Combined.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test_challenge1_Option2.jpg" width="150" height1="100">|
<img src="./output_images/perspective_transform/test_challenge2.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test_challenge2_SchannelOnly.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test_challenge2_FullColorTransf.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test_challenge2_Combined.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test_challenge2_Option2.jpg" width="150" height1="100">|
<img src="./output_images/perspective_transform/test_challenge3.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test_challenge3_SchannelOnly.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test_challenge3_FullColorTransf.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test_challenge3_Combined.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test_challenge3_Option2.jpg" width="150" height1="100">|
<img src="./output_images/perspective_transform/test_challenge4.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test_challenge4_SchannelOnly.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test_challenge4_FullColorTransf.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test_challenge4_Combined.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/test_challenge4_Option2.jpg" width="150" height1="100">|

## Challenge Video Color Transform
| Perspective transform | S channel Only |Sobel + thresholds | Combined | Yellow & White mask only|
------------ | ------------- | ------------- | ------------- | -------------
<img src="./output_images/perspective_transform/hard_challenge0.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/hard_challenge0_SchannelOnly.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/hard_challenge0_FullColorTransf.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/hard_challenge0_Combined.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/hard_challenge0_Option2.jpg" width="150" height1="100">|
<img src="./output_images/perspective_transform/hard_challenge1.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/hard_challenge1_SchannelOnly.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/hard_challenge1_FullColorTransf.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/hard_challenge1_Combined.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/hard_challenge1_Option2.jpg" width="150" height1="100">|
<img src="./output_images/perspective_transform/hard_challenge2.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/hard_challenge2_SchannelOnly.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/hard_challenge2_FullColorTransf.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/hard_challenge2_Combined.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/hard_challenge2_Option2.jpg" width="150" height1="100">|
<img src="./output_images/perspective_transform/hard_challenge3.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/hard_challenge3_SchannelOnly.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/hard_challenge3_FullColorTransf.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/hard_challenge3_Combined.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/hard_challenge3_Option2.jpg" width="150" height1="100">|
<img src="./output_images/perspective_transform/hard_challenge4.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/hard_challenge4_SchannelOnly.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/hard_challenge4_FullColorTransf.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/hard_challenge4_Combined.jpg" width="150" height1="100">|<img src="./output_images/perspective_transform/hard_challenge4_Option2.jpg" width="150" height1="100">|
