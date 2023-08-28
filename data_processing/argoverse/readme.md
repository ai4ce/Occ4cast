# Install dependencies
Install [argoverse-api](https://github.com/argoverse/argoverse-api). You don't need any of the optional installation(mayavi,ffmpeg...etc.) to run this code.

# Download argoverse tracking dataset
Download Argoverse 3D Tracking v1.1 from [here](https://www.argoverse.org/av1.html#tracking-link). After unzipping, there should be these 5 folders: 'train1','train2','train3','train4','val'

# Run this code
```
python transform.py --dataset_dir ..../train1 --output_path /path/to/the/output
python transform.py --dataset_dir ..../train2 --output_path /path/to/the/output
python transform.py --dataset_dir ..../train3 --output_path /path/to/the/output
python transform.py --dataset_dir ..../train4 --output_path /path/to/the/output
python transform.py --dataset_dir ..../val --output_path /path/to/the/output
```

