cd ~/projects/experiments_hinagata/exp/sample
sbatch --partition=a6000_ada --gres=gpu:2 --cpus-per-task=8 -J test << "EOF"
#!/bin/bash
set -e
source ~/projects/experiments_hinagata/.venv/bin/activate
./main.py \
    --configs \
        configs/base_config.yaml \
        configs/cifar10.yaml \
        configs/resnet18.yaml \
    --save-freq 3
EOF


