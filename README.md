# Medical Images Stitching

C++ implementation for Medical Image Stitching.

## Introduction

In the Automated Medical Image Scanning, we have a significant issue with the camera position in its path, thus the captured images is so likely to have overlapping fields with horizental and vertical adjacent images. My friends and I has developed a program in C++(due to its speed) which exports a DLL to be used in any project(I write a example for C#).

#### Overlapping image samples before stitching:

Row 7 - Columns 26 : 30:
<center>
<img src="https://github.com/AmirShahid/Stitching/blob/master/images/img_7_26.jpeg" width="153" height="128" /><img src="https://github.com/AmirShahid/Stitching/blob/master/images/img_7_27.jpeg" width="153" height="128" /><img src="https://github.com/AmirShahid/Stitching/blob/master/images/img_7_28.jpeg" width="153" height="128" /><img src="https://github.com/AmirShahid/Stitching/blob/master/images/img_7_29.jpeg" width="153" height="128" /><img src="https://github.com/AmirShahid/Stitching/blob/master/images/img_7_30.jpeg" width="153" height="128" />
</center>
Row 8 - Columns 26 : 30:
<center>
<img src="https://github.com/AmirShahid/Stitching/blob/master/images/img_8_26.jpeg" width="153" height="128" /><img src="https://github.com/AmirShahid/Stitching/blob/master/images/img_8_27.jpeg" width="153" height="128" /><img src="https://github.com/AmirShahid/Stitching/blob/master/images/img_8_28.jpeg" width="153" height="128" /><img src="https://github.com/AmirShahid/Stitching/blob/master/images/img_8_29.jpeg" width="153" height="128" /><img src="https://github.com/AmirShahid/Stitching/blob/master/images/img_8_30.jpeg" width="153" height="128" />
</center>
## Requirements

* [OpenCV](https://github.com/opencv/opencv) 3.4.3
* [Xtensor](https://github.com/xtensor-stack/xtensor) 0.20.5

## Data 

We've made our own datasets from our Basler camera.

## Questions
Please add an issue if you have any questions
