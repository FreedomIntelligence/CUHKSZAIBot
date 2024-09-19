# need change 4 place
# Please set the wandb key in the python file (e.g trainDataConstruct.py)

cd ./modelTrain
mkdir wandb_logs

experiment_name=/your/experiment/name
log_folder="./logs/${experiment_name}"
mkdir -p $log_folder
log_name=$(date +"%m-%d_%H-%M").log


python trainDataConstruct.py \
--data_path ./path/to/your/mixTrain/data \
--model_path /your/path/to/model \
--wandb_log ./wandb_logs \
--experiment_name ${experiment_name} \
--save_path ./data/mixTrain > ${log_folder}/$log_name 2>&1 &