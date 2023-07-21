# Install dependencies
1. Open `requirements.txt` and replace `XXX` with the cuda version on your system. For example, if your cuda version is `11.8`, then replace `cuXXX` with `cu118`.
2. Run `pip install -r requirements.txt`

# Run the code
1. Setup parameters in `run_convert_lyft.sh` then run it by executing `./run_convert_lyft.sh`
2. Setup parameters in `run_ray_model.sh` then run it by executing `./run_ray_model.sh`

# Data folder structure format
The data format after processing should be as follows:
```
dataset/lyft/
├── train

|   ├── calib
|   |   ├── 0000 # scene index
|   |   |   ├── cam_extrinsics.npz
|   |   |   ├── cam_intrinsics.npz
|   |   ├── ...

|   ├── image 
|   |   ├── 0000 # scene index
|   |   |   ├── 0000_CAM_FRONT.jpg
|   |   |   ├── 0000_CAM_FRONT_LEFT.jpg
|   |   |   ├── ...
|   |   |   ├── 0001_CAM_FRONT.jpg
|   |   |   ├── ...
|   |   ├── ...

|   ├── point_cloud
|   |   ├── 0000 # scene index
|   |   |   ├── 0000_point.npz
|   |   |   ├── 0000_label.npz
|   |   |   ├── 0000_pose.npz
|   |   |   ├── 0001_point.npz
|   |   |   ├── ...
|   |   ├── ...

|   ├── voxel
|   |   ├── 0000 # scene index
|   |   |   ├── 0010.npz # Start from 10th frame
|   |   |   ├── 0011.npz
|   |   |   ├── ...
|   |   ├── ...

|   ├── vis # Optional
```

# `.npz` file structure
Differnt `.npz` files have structure as follows:
- `cam_extrinsics.npz`: 
```
CAM_FRONT: 4x4 matrix
CAM_FRONT_LEFT: 4x4 matrix
CAM_FRONT_RIGHT: 4x4 matrix
CAM_BACK: 4x4 matrix
CAM_BACK_LEFT: 4x4 matrix
CAM_BACK_RIGHT: 4x4 matrix
```
- `cam_intrinsics.npz`: 
```
CAM_FRONT: 3x3 matrix
...
```
- `*_point.npz`: 
```
i_LIDAR_TOP: Nx3 matrix
i_LIDAR_FRONT_LEFT: Nx3 matrix
i_LIDAR_FRONT_RIGHT: Nx3 matrix
```
Note: `i` starts from `*-10` and ends at `*+10` for each scene, i.e. when `*=10`, `i` takes values from `[0, 1, ..., 20]`
- `*_label.npz`: 
```
i_LIDAR_TOP: Nx1 matrix
...
```
- `*_pose.npz`: 
```
i_LIDAR_TOP: 4x4 matrix
...
```
- '*.npz' (in `voxel` folder): 
```
input: (B+1)x512x256x32 matrix # B is the number of previous frames, default is 10
label: (B+1)x512x256x32 matrix # B is the number of future frames, default is 10
invalid: (B+1)x512x256x32 matrix # same as label
```
For label details, refer to the [config file](config/class_map.yaml)