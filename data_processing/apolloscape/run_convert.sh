# Data path
DATA_PATH=/vast/xl3136/apolloscape

# Output path
OUTPUT_PATH=/vast/xl3136/apolloscape_kitti

# Start and end scene index
START=8
END=9

# Number of frames to aggregrate
A_PRE=70
A_POST=70

# Number of frames to predict
P_PRE=10
P_POST=10

python convert_apollo.py -d $DATA_PATH -o $OUTPUT_PATH -s $START -e $END --a_pre $A_PRE --a_post $A_POST --p_pre $P_PRE --p_post $P_POST