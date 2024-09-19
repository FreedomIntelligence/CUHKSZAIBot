#!/bin/bash
source /root/anaconda3/etc/profile.d/conda.sh
conda activate webui
cd /mnt/nvme2n1

export PORT=8082

nohup open-webui/backend/start.sh > output.log 2>&1 &

echo "Script executed successfully, backend started in background."

