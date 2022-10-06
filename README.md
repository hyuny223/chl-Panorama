# chl-stitching
It is developing.  
  
The goal of this project is to make a auto-stitching program without OpenCV except cv::Mat and cv::imshow().  
To achieve this, some functions are needed.  
  
## 1. Parser
In order to load images, it would be good to make parser using "filesystem" in C++.  
## 2. Cylindrical Warping
To reduce distortion, cylindrical warping would be good choice.
## 3. RANSAC
To remove outliers, this is essential technique.
## 4. Optimization
To compute accurate homography matrix, not only RANSAC, but also Optimization is essential! In my case, I implemented Levenberg Marquardt algorithm.  
So far, it's only for homography.
## 5. Image Warping
To stitch images, we should project images on canvas based on homography matrix. To prevent holes during warping, interpolation is needed. In my cases, I used bilinear interpoltaion.
## 6. Blending
develpoing...
## 7. Graph Cut
develpoing...
