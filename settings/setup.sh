# conda create -n omniphysgs python=3.11.9
# conda activate omniphysgs
# conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r settings/requirements.txt

pip install gaussian3d/submodules/diff-gaussian-rasterization/
pip install gaussian3d/submodules/simple-knn/
