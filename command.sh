conda create --prefix /home/users/ntu/yongxias/scratch/envs/dual python=3.8
cd /home/users/ntu/yongxias/miniconda3/envs
ln -s /home/users/ntu/yongxias/scratch/envs/dual ./
conda activate dual

conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirement.txt

cd scratch/lilin_projects/dual_system

qsub  -I -l select=1:ngpus=1 -P personal-keqichen -l walltime=1:00:00
module load cuda/12.2.2














sftp yongxias@aspire2antu.nscc.sg

ls 
pwd
lls 
put/get filename

put/get -r folder

