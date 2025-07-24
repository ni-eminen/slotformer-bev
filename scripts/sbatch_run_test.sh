#!/bin/bash

# SBATCH file can't directly take command args
# as a workaround, I first use a sh script to read in args
# and then create a new .slrm file for SBATCH execution

#######################################################################
# An example usage:
#     GPUS=1 CPUS_PER_GPU=8 MEM_PER_CPU=5 QOS=normal ./scripts/sbatch_run.sh rtx6000 \
#         test-sbatch test.py ddp --params params.py --fp16 --ddp --cudnn
#######################################################################

# read args from command line
GPUS=${GPUS:-1}
CPUS_PER_GPU=${CPUS_PER_GPU:-8}
MEM_PER_CPU=${MEM_PER_CPU:-5}
QOS=${QOS:-normal}
TIME=${TIME:-0}

PY_ARGS=${@:5}
PARTITION=$1
JOB_NAME=$2
PY_FILE=$3
DDP=$4

SLRM_NAME="${JOB_NAME/\//"_"}"
LOG_DIR=checkpoint/"$(basename -- $JOB_NAME)"
DATETIME=$(date "+%Y-%m-%d_%H:%M:%S")
LOG_FILE=$LOG_DIR/${DATETIME}.log
CPUS_PER_TASK=$((GPUS * CPUS_PER_GPU))
WANDB_API_KEY=d5b53100f4a6048f5420ed067837cf6625703d34
# set up log output folder
mkdir -p $LOG_DIR

# python runner for DDP
if [[ $DDP == "ddp" ]];
then
  PORT=$((29501 + $RANDOM % 100))  # randomly select a port
  PYTHON="python -u -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT"
else
  PYTHON="python -u"
fi

# write to new file
echo "#!/bin/bash

# set up SBATCH args
#SBATCH --account=project_2014099
#SBATCH --job-name=$SLRM_NAME
#SBATCH --partition=$PARTITION                       # self-explanatory, set to your preference (e.g. gpu or cpu on MaRS, p100, t4, or cpu on Vaughan)
#SBATCH --cpus-per-task=10               # self-explanatory, set to your preference
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=0                # self-explanatory, set to your preference
#SBATCH --gres=gpu:a100:2                             # NOTE: you need a GPU for CUDA support; self-explanatory, set to your preference 
#SBATCH --time=00:15:00                                 # running time limit, 0 as unlimited

# log some necessary environment params
echo \$SLURM_JOB_ID >> $LOG_FILE                      # log the job id
echo \$SLURM_JOB_PARTITION >> $LOG_FILE               # log the job partition

echo $CONDA_PREFIX >> $LOG_FILE                      # log the active conda environment 

python --version >> $LOG_FILE                        # log Python version
gcc --version >> $LOG_FILE                           # log GCC version
nvcc --version >> $LOG_FILE                          # log NVCC version

# run python file
$PYTHON $PY_FILE $PY_ARGS >> $LOG_FILE                # the script above, with its standard output appended log file

" >> ./run-${SLRM_NAME}.slrm

# run the created file
sbatch run-${SLRM_NAME}.slrm

# delete it
sleep 0.1
rm -f run-${SLRM_NAME}.slrm
