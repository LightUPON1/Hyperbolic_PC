CUDA_VISIBLE_DEVICES=$1 python main.py --config cfgs/svm/modelnet40.yaml \
--svm --exp_name $2 --ckpts $3

#CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/svm/modelnet40.yaml --svm --exp_name base_meru/meru --ckpts best
