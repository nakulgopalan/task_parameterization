#!/bin/bash
###source /home/ng/workspace/task_parameterization/task_param_env/bin/activate

source /home/ngopalan/workspace/task_param_expts/task_param_env/bin/activate
let num=200
line_num=`expr $SGE_TASK_ID + $num`
PARAMS="$(sed -n "$line_num p" /home/ngopalan/workspace/task_param_expts/task_parameterization/tau.txt)"
echo $line_num
echo $PARAMS
MASS="$(cut -d' ' -f1 <<<$PARAMS)"
LENGTH="$(cut -d' ' -f2 <<<$PARAMS)"
echo $MASS
echo $LENGTH
python /home/ngopalan/workspace/task_param_expts/task_parameterization/reinforce.py --mass $MASS --length $LENGTH
