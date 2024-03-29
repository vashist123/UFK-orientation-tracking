# UFK-orientation-tracking
7 DOF UKF for tracking orientation of a drone

The ground truth orientation is captured using the vicon and is given in the `vicon` folder. he raw data of the IMU is given in the `imu` folder.

The green line represents the ground truth orientation given by the vicon. The blue line represents the mean of the estimated orientation while the red bounds give the standard deviation from the mean.

## Dataset 1
<img src = "https://github.com/vashist123/UFK-orientation-tracking/blob/main/images/with_std_1.png" width="700" height="500">

## Dataset 2
<img src = "https://github.com/vashist123/UFK-orientation-tracking/blob/main/images/with_std_2.png" width="700" height="500">

## Reference
> E. Kraft, "A quaternion-based unscented Kalman filter for orientation tracking," Sixth International Conference of Information Fusion, 2003. Proceedings of the, Cairns, Queensland, Australia, 2003, pp. 47-54, doi: 10.1109/ICIF.2003.177425.
