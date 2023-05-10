# Images Stitching <sub>***Part of our Automated Slide Scanner***</sub>

Multi-threaded C++ implementation for Image Stitching. 

## Introduction

In the Automated Medical Image Scanning, we have a significant issue with the moving camera position on the slide, thus the captured images is so likely to have overlapping fields with horizental and vertical adjacent images. We have developed a program in C++(due to its performance) which exports a DLL to be used in any project(an example for C# is under Stitching_CSHARP) to concat images regarding to the Matching Common Templates among adjacent overlapping images.

#### *Overlapping image samples before stitching*:

*Row 7 : 8 - Columns 26 : 30:*
<p align="center">
   <img src="https://github.com/AmirShahid/Stitching/blob/master/images/img_7_26.jpeg" width="153" height="128" /><img  src="https://github.com/AmirShahid/Stitching/blob/master/images/img_7_27.jpeg" width="153" height="128" /><img    src="https://github.com/AmirShahid/Stitching/blob/master/images/img_7_28.jpeg" width="153" height="128" /><img    src="https://github.com/AmirShahid/Stitching/blob/master/images/img_7_29.jpeg" width="153" height="128" /><img    src="https://github.com/AmirShahid/Stitching/blob/master/images/img_7_30.jpeg" width="153" height="128" />
   <img src="https://github.com/AmirShahid/Stitching/blob/master/images/img_8_26.jpeg" width="153" height="128" /><img  src="https://github.com/AmirShahid/Stitching/blob/master/images/img_8_27.jpeg" width="153" height="128" /><img    src="https://github.com/AmirShahid/Stitching/blob/master/images/img_8_28.jpeg" width="153" height="128" /><img    src="https://github.com/AmirShahid/Stitching/blob/master/images/img_8_29.jpeg" width="153" height="128" /><img  src="https://github.com/AmirShahid/Stitching/blob/master/images/img_8_30.jpeg" width="153" height="128" />
</p>

## Usage

1. Clone to the repository:
   ```
   git clone https://github.com/AmirShahid/Stitching.git
   ```

2. Set Stitching/StitchConfig.json parameters as you want it can run on RAM to be faster. if you set load_from_disk you should address your data as said in the Data section.

3. Build the C++ Project to export a DLL including its dependencies like jpeg62.dll,opencv_imgproc343.dll, ...

_(You can use it without exporting DLL in C++ projects like in Stitching/Main.cpp)_

4. Put DLL files beside your project and whenever camera iterate a row in lamel call _get_lr_shifts_ then save the output shifts

5. Finally when the whole lamel has been scanned call _Stitch_all_ to get an array including the whole lamel image in diffrent zoom levels.

### Output Structure 
The output are the small non-overlapping tiles with diffrent details like google map tilesets and can be read through [leaflet](https://rstudio.github.io/leaflet/). the tiles in a specific zoom level can be concated to make a full lamel image. you can see a 40x40 output for 7 zoom leves [here](https://drive.google.com/open?id=10pyts1j4yTH7hfwQ6NXlntlE_68Vex9-), this is a example for the top zoom level:
<p align="center">
<img  src="https://github.com/AmirShahid/Stitching/blob/master/images/stitch_output.jpg" width="776" height="770"/>
</p>

for more guidance you can see program.cs under Stitching_CSHARP.
## Requirements

* [OpenCV](https://github.com/opencv/opencv) 3.4.3
* [Xtensor](https://github.com/xtensor-stack/xtensor) 0.20.5
* [Leaflet](https://rstudio.github.io/leaflet/) (Optional for visualization)

## Data 

We've made our own datasets with our Basler camera. you can add your dataset with setting data_dir parameter in StitchConfig.json and the images names can be like *{pref}\_{row_number}\_{column_number}.{ext}* where pref and ext should set through StitchConfig.

## Questions
Please add an issue if you have any question
