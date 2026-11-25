CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/full/fewshot.yaml --finetune_model \
--exp_name meru_woHyploss \
--ckpts ~/CodeSpace/Hyperbolic_PC/experiments/base_meru/pretrain/meru_woHyploss/ckpt-best.pth \
--way 5 \
--shot 20 \
--fold 0

CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/full/fewshot.yaml --finetune_model \
--exp_name meru_woHyploss \
--ckpts ~/CodeSpace/Hyperbolic_PC/experiments/base_meru/pretrain/meru_woHyploss/ckpt-best.pth \
--way 5 \
--shot 20 \
--fold 0

CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/full/fewshot.yaml --finetune_model \
--exp_name meru_woHyploss \
--ckpts ~/CodeSpace/Hyperbolic_PC/experiments/base_meru/pretrain/meru_woHyploss/ckpt-best.pth \
--way 5 \
--shot 20 \
--fold 0
