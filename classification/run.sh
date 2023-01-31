python train_down_ext.py --gpu_id 0 --model Vision_Transformer --learning_rate 0.00008 --model_name ViT_large_r50_s32_384_mask_4 --batch_size 8
python train_down_ext.py --gpu_id 2 --model Vision_Transformer --learning_rate 0.0001 --model_name ViT_large_r50_s32_384_mask_4 --batch_size 8
python train_down_ext.py --gpu_id 3 --model Vision_Transformer --learning_rate 0.00009 --model_name ViT_large_r50_s32_384_mask_4 --batch_size 8
python train_down_ext.py --gpu_id 1 --model Vision_Transformer --learning_rate 0.0001 --model_name ViT_large_r50_s32_384_mask_5 --batch_size 8
python train_down_com.py --gpu_id 0 --model Vision_Transformer --learning_rate 0.0001 --model_name ViT_large_r50_s32_384_mask_5 --batch_size 8
python train_down_com.py --gpu_id 1 --model Vision_Transformer --learning_rate 0.00009 --model_name ViT_large_r50_s32_384_mask_5 --batch_size 8



다음 실험







