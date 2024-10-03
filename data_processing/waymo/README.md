# Data folder structure format
Waymo data format:
```
dataset/waymo/
├── calib
|   ├── 0000 # scene index
|   |   ├── CAM_FRONT_CENTER_extrinsic.bin
|   |   ├── CAM_FRONT_CENTER_intrinsic.bin
|   |   ├── ...
|   |   ├── CAM_SIDE_RIGHT_extrinsic.bin
|   |   ├── CAM_SIDE_RIGHT_intrinsic.bin
|   ├── ...
|   ├── 0999
├── voxel
|   ├── 0000 # scene index
|   |   ├── 0005.npz # Start from 5th frame
|   |   ├── 0006.npz
|   |   ├── ...
|   ├── ...
|   ├── 0999
├── image
|   ├── 0000 # scene index
|   |   ├── 0000_CAM_FRONT_CENTER.jpg
|   |   ├── 0000_CAM_FRONT_LEFT.jpg
|   |   ├── ...
|   |   ├── 0001_CAM_FRONT_CENTER.jpg
|   |   ├── ...
|   ├── ...
|   ├── 0999
├── image_downsample
|   ├── 1_2 # 50% downsampled in size
|   |   |── 0000
|   |   |   ├── 0000_CAM_FRONT_CENTER.jpg
|   |   |   ├── 0000_CAM_FRONT_LEFT.jpg
|   |   |   ├── ...
|   |   |   ├── 0001_CAM_FRONT_CENTER.jpg
|   |   |   ├── ...
|   |   |── ...
|   |   |── 0999
|   ├── 1_4#  25% downsampled in size
|   |   |── 0000
|   |   |   ├── 0000_CAM_FRONT_CENTER.jpg
|   |   |   ├── 0000_CAM_FRONT_LEFT.jpg
|   |   |   ├── ...
|   |   |   ├── 0001_CAM_FRONT_CENTER.jpg
|   |   |   ├── ...
|   |   |── ...
|   |   |── 0999
```
# calib file
```
CAMERA_NAME_extrinsic.bin: 4x4 matrix
CAMERA_NAME_intrinsic.bin: 3x3 matrix
```

# `.npz` file structure
Differnt `.npz` files have structure as follows:
- '*.npz' (in `voxel` folder): 
```
input: (B+1)x512x256x32 matrix # B is the number of previous frames, default is 10
label: (B+1)x512x256x32 matrix # B is the number of future frames, default is 10
invalid: (B+1)x512x256x32 matrix # same as label
```
For label details, refer to the [config file](data_processing/waymo/waymo_class_map.yaml)
