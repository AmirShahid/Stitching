# Medical Images Stitching

C++ implementation for Medical Image Stitching.

## Introduction

In the Automated Medical Image Scanning, we have a significant issue with the camera position in its path, thus the captured images is so likely to have overlapping fields with horizental and vertical adjacent images. My friends and I has developed a program in C++(due to its speed) which exports a DLL to be used in any project(I write a example for C#).

Overlapping Image Samples:

Row 7 - Columns 21 : 32:
<img src=""https://github.com/AmirShahid/Stitching/blob/master/images/img_7_21.jpeg"" width="153" height="128" />
<img src=""https://github.com/AmirShahid/Stitching/blob/master/images/img_7_22.jpeg"" width="153" height="128" />
<img src=""https://github.com/AmirShahid/Stitching/blob/master/images/img_7_23.jpeg"" width="153" height="128" />
<img src=""https://github.com/AmirShahid/Stitching/blob/master/images/img_7_24.jpeg"" width="153" height="128" />
<img src=""https://github.com/AmirShahid/Stitching/blob/master/images/img_7_25.jpeg"" width="153" height="128" />
<img src=""https://github.com/AmirShahid/Stitching/blob/master/images/img_7_26.jpeg"" width="153" height="128" />
<img src=""https://github.com/AmirShahid/Stitching/blob/master/images/img_7_27.jpeg"" width="153" height="128" />
<img src=""https://github.com/AmirShahid/Stitching/blob/master/images/img_7_28.jpeg"" width="153" height="128" />
<img src=""https://github.com/AmirShahid/Stitching/blob/master/images/img_7_29.jpeg"" width="153" height="128" />
<img src=""https://github.com/AmirShahid/Stitching/blob/master/images/img_7_30.jpeg"" width="153" height="128" />
<img src=""https://github.com/AmirShahid/Stitching/blob/master/images/img_7_31.jpeg"" width="153" height="128" />
<img src=""https://github.com/AmirShahid/Stitching/blob/master/images/img_7_32.jpeg"" width="153" height="128" />

## Requirements

* [OpenCV](https://github.com/opencv/opencv) 3.4.3
* [Xtensor](https://github.com/xtensor-stack/xtensor) 0.20.5

## Data 

We've made our own datasets from our Basler camera.

## Questions
Please add an issue if you have any questions
