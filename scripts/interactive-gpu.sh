# srun --time=01:00:00 --partition=debug-cpu --cpus-per-task=20 --pty /bin/bash

srun --time=01:00:00 --partition=debug-gpu --cpus-per-task=20 --gres=gpu:volta:1 --pty /bin/bash
