# Characterising-resource-usage-for-RNN

Commands:
sudo apt-get install gnupg-curl

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.0.130-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu1604_10.0.130-1_amd64.deb

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub

sudo apt-get update

wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb

sudo apt install ./nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
sudo apt-get update

sudo mkdir /usr/lib/nvidia
sudo apt-get install --no-install-recommends nvidia-418

sudo apt-get install --no-install-recommends cuda-10-0 libcudnn7=7.6.2.24-1+cuda10.0 libcudnn7-dev=7.6.2.24-1+cuda10.0

sudo pip3 install keras

Reference Links: 
1. For GCP set up: https://medium.com/datadriveninvestor/complete-step-by-step-guide-of-keras-transfer-learning-with-gpu-on-google-cloud-platform-ed21e33e0b1d

2. For tensorflow GPU: https://www.tensorflow.org/install/gpu

