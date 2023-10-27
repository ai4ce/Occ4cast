# Data path
DATA_PATH=/media/xinhao/DATA/nuscenes_demo/

# Output path
OUTPUT_PATH=/media/xinhao/DATA/nusc_kitti_demo

# Start and end scene index
START=0

# Number of frames to aggregrate
A_PRE=70
A_POST=70

# Number of frames to predict
P_PRE=10
P_POST=10

# python convert_nuscenes.py -b 0 -d $DATA_PATH -o $OUTPUT_PATH -s $START --a_pre $A_PRE --a_post $A_POST --p_pre $P_PRE --p_post $P_POST
parallel 'python convert_nuscenes.py -b {}' -d $DATA_PATH -o $OUTPUT_PATH -s $START --a_pre $A_PRE --a_post $A_POST --p_pre $P_PRE --p_post $P_POST ::: {0..9}