from models.meru.meru import load_meru

train_config='/home/lyj/CodeSpace/Hyperbolic_PC/models/meru/config_train_meru_vit_b.py'
checkpoint_path = '/home/lyj/CodeSpace/Hyperbolic_PC/models/checkpoint/meru_vit_b.pth'

MERU_pretrained_model = load_meru(train_config, checkpoint_path)
# VisionTransformer, TextTransformer, fun_VisionTransformer, fun_TextTransformer = load_meru()
