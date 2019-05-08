# Copy of HMR

## Prerequisites
- [HMR paper](https://github.com/akanazawa/hmr)
- [CMU OpenPose paper](https://github.com/akanazawa/hmr)

## Video to Video(not real-time)

- `bash vtois.sh <Video Name> <FPS>` will save images in `images_from_video/`
- `python 2d_pose_estimation.py <Images Dir Name>` will save  2d joints results in `jsons_from_video/` by using [OpenPose paper](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
   > in order to compute the right scale and bbox center
   >

- `bash 3d_pose_estimation.sh <Images Dir Name>` will save 3d mesh results in`meshes_from_video/` by using [HMR paper](https://github.com/akanazawa/hmr)
- `bash istov.sh <Meshes Dir Name> <FPS>` will generate final mesh-video

![sample2](GIF.gif "sample2")
