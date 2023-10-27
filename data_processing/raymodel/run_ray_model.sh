START=0
END=1
A_PRE=70
A_POST=70
P_PRE=10
P_POST=10
INPUT=/media/xinhao/DATA/nusc_kitti_demo

python ray_model.py -v -i $INPUT -s $START -e $END --a_pre $A_PRE --a_post $A_POST --p_pre $P_PRE --p_post $P_POST