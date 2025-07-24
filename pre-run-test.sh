GPUS=2 CPUS_PER_GPU=8 MEM_PER_CPU=5 TIME=00:14:00 QOS=gpu \
	    ./scripts/sbatch_run_test.sh gputest slotformer \
	        scripts/train.py ddp --task bev_slots --params slotformer/bev_slots/configs/savi_bev_params_test.py --weight /scratch/project_2014099/previous_runs/town03/lts.pth --fp16 --ddp --cudnn


