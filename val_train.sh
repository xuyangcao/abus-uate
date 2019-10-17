python ./train.py --save ./work/10_17_super_dense121_maxval_3_lr_1e-4_drop_0.3_finetune_wd_1e-4 --sample_k 4873 --max_val 3 --train_method super --ngpu 1 --gpu_idx 0 --batchsize 16 --arch dense121 --drop_rate 0.3 --max_epochs 40 --n_epochs 100 # 512x128 

# the following code validate the optimize input size and batch size, we choose input size 512x128, and batch size 10-16
#python ./train.py --save ./work/10_16_super_dense121_maxval_3_lr_1e-4_drop_0.3_finetune_wd_1e-4_4873 --sample_k 4873 --max_val 3 --train_method super --ngpu 2 --gpu_idx 0,1 --batchsize 10 --arch dense --drop_rate 0.3 --max_epochs 40 --n_epochs 100 # 512x128 
#python ./train.py --save ./work/10_15_super_dense121_maxval_3_lr_1e-4_drop_0.3_finetune_wd_1e-4_4873 --sample_k 4873 --max_val 3 --train_method super --ngpu 1 --gpu_idx 2,3 --batchsize 10 --arch dense --drop_rate 0.3 --max_epochs 40 --n_epochs 100 # 512x128 
#python ./train.py --save ./work/10_14_super_dense121_maxval_3_lr_1e-4_drop_0.3_finetune_wd_1e-4_4873 --sample_k 4873 --max_val 3 --train_method super --ngpu 1 --gpu_idx 2,3 --batchsize 10 --arch dense --drop_rate 0.3 --max_epochs 40 --n_epochs 100 # 224x224 
