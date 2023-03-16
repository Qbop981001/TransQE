
create -n qbp python=3.7
pip install transformers==3.5.1
pip uninstall protobuf
pip install protobuf==3.20.0
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch