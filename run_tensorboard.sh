#!/bin/bash
#SBATCH --time=0-12:00                  # How long you want to run for
#SBATCH --account=def-amw8          # The account you want to bill, if you have access to Martha's it's rrg-whitem or def-whitem. I don't know what Adam's is.
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --job-name=relgan-tensorboard
#SBATCH --output=/home/jcoblin/projects/def-amw8/jcoblin/RelGAN/logs/tensorboard/%x-output-tensorboard-%j.log
#SBATCH --error=/home/jcoblin/projects/def-amw8/jcoblin/RelGAN/logs/tensorboard/%x-error-tensorboard-%j.log

source .venv/relgan/bin/activate
# module load python/3.7

tensorboard --logdir='/home/jcoblin/projects/def-amw8/jcoblin/RelGAN/oracle/experiments/out/20221201_1449/oracle/oracle_rmc_vanilla_hinge_adam_bs64_sl20_sn0_dec0_ad-no_npre150_nadv3000_ms1_hs256_nh2_ds5_dlr1e-4_glr1e-4_tem1_demb64_nrep64_hdim32_sd99_sparseTrue/tf_logs' --host 0.0.0.0 --port=6009
