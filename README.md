# Images Stitching

Multi-threaded C++ implementation for Image Stitching.

## Introduction

In the Automated Medical Image Scanning, we have a significant issue with the camera position through the path on the slide, thus the captured images is so likely to have overlapping fields with horizental and vertical adjacent images. My friends and I has developed a program in C++(due to its speed) which exports a DLL to be used in any project(I write a example for C#).

#### *Overlapping image samples before stitching*:

*Row 7 : 8 - Columns 26 : 30:*

   <img src="https://github.com/AmirShahid/Stitching/blob/master/images/img_7_26.jpeg" width="153" height="128" /><img  src="https://github.com/AmirShahid/Stitching/blob/master/images/img_7_27.jpeg" width="153" height="128" /><img    src="https://github.com/AmirShahid/Stitching/blob/master/images/img_7_28.jpeg" width="153" height="128" /><img    src="https://github.com/AmirShahid/Stitching/blob/master/images/img_7_29.jpeg" width="153" height="128" /><img    src="https://github.com/AmirShahid/Stitching/blob/master/images/img_7_30.jpeg" width="153" height="128" />
   <img src="https://github.com/AmirShahid/Stitching/blob/master/images/img_8_26.jpeg" width="153" height="128" /><img  src="https://github.com/AmirShahid/Stitching/blob/master/images/img_8_27.jpeg" width="153" height="128" /><img    src="https://github.com/AmirShahid/Stitching/blob/master/images/img_8_28.jpeg" width="153" height="128" /><img    src="https://github.com/AmirShahid/Stitching/blob/master/images/img_8_29.jpeg" width="153" height="128" /><img  src="https://github.com/AmirShahid/Stitching/blob/master/images/img_8_30.jpeg" width="153" height="128" />

## Usage

1. Clone to the repository:
   ```
   git clone https://github.com/AmirShahid/Stitching.git
   ```

2. Set Stitching/StitchConfig.json parameters as you want it can run on RAM to be faster. if you set load_from_disk you should address your data as said in the Data section.


3. The output are the small non-overlapping images which can be concated to make a full lamel image like below

<img  src="https://github.com/AmirShahid/Stitching/blob/master/images/stitch_output.jpg" width="776" height="770"/>


for more guidance you can see program.cs in Stitching_CSHARP.
## Requirements

* [OpenCV](https://github.com/opencv/opencv) 3.4.3
* [Xtensor](https://github.com/xtensor-stack/xtensor) 0.20.5

## Data 

We've made our own datasets with our Basler camera. you can add your dataset with setting data_dir parameter in StitchConfig,json and and the images names can be like *{pref}\_{row_number}\_{column_number}.{ext}* where pref and ext should set through StitchConfig.

## Questions
Please add an issue if you have any questions
