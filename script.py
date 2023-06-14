#! /bin/bash
#SBATCH --job-name=my_job1_test    # Job name
#SBATCH -N 1                  # 1 nœud
#SBATCH --exclusive           # Le nœud sera entièrement dédié à notre job, pas de partage de ressources
#SBATCH -t 05:00:00           # Le job sera tué au bout de 5h
#SBATCH --mail-type=END       # Réception d'un mail à la fin du job
#SBATCH --mail-user=tokparosali@gmail.com

echo "Running plot script on a single GPU core"

./vdc_bilstm.py
