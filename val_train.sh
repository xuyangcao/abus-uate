python train_s4GAN.py --save ./work/gan_task2/s4gan_baseline
#python train_mtvat.py --save ./work/gan_task2/mtvat


#python train_s4GAN.py --save ./work/gan/s4gan_baseline_adam_5000_0.65_th0.4 --threshold_st 0.4
#python train_s4GAN.py --save ./work/gan/s4gan_baseline_adam_8000_0.5_th0.4 --threshold_st 0.4
#python train_s4GAN.py --save ./work/gan/s4gan_baseline_adam_8000_0.5

#python train_s4GAN.py --save ./work/gan/s4gan_baseline_withgd --threshold_st 0.6 -k 100 --lambda_fm 0.1 --learning_rate 0.0001 --learning_rate_D 0.000001 --lambda_adv 0.01 --in_channels_d 3 --weight_decay 0.0001
#python train_s4GAN.py --save ./work/gan/s4gan_baseline --threshold_st 0.6 -k 100 --lambda_fm 0.1 --learning_rate 0.0001 --learning_rate_D 0.000001 --lambda_adv 0.01 --in_channels_d 3 --weight_decay 0.0001
#python train_s4GAN.py --save ./work/gan/s4gan_baseline_linear --threshold_st 0.6 -k 100 --lambda_fm 0.1 --learning_rate 0.0001 --learning_rate_D 0.000001 --lambda_adv 0.1 --in_channels_d 3 --weight_decay 0.0001
#python train_s4GAN.py --save ./work/gan/s4gan_baseline_nonlinear --threshold_st 0.6 -k 100 --lambda_fm 0.1 --learning_rate 0.0001 --learning_rate_D 0.000001 --lambda_adv 0.1 --in_channels_d 3 --weight_decay 0.0001

#python train_s4GAN.py --save ./work/gan/s4gan_lr1e-3_wd1e-4_lrd1e-6_linear0.9_every1000_withgrid --threshold_st 0.6 -k 100 --lambda_fm 0.1 --learning_rate 0.001 --learning_rate_D 0.000001 --lambda_adv 0.1 --in_channels_d 3 --weight_decay 0.0001
#python train_s4GAN.py --save ./work/gan/s4gan_lr1e-3_wd1e-4_lrd1e-6_linear0.8_every2500_withgrid --threshold_st 0.6 -k 100 --lambda_fm 0.1 --learning_rate 0.001 --learning_rate_D 0.000001 --lambda_adv 0.1 --in_channels_d 3 --weight_decay 0.0001
#python train_s4GAN.py --save ./work/gan/s4gan_lr1e-3_wd1e-4_lrd1e-6_linear0.8_every2500 --threshold_st 0.6 -k 100 --lambda_fm 0.1 --learning_rate 0.001 --learning_rate_D 0.000001 --lambda_adv 0.1 --in_channels_d 3 --weight_decay 0.0001
#python train_s4GAN.py --save ./work/gan/s4gan_lr1e-3_wd1e-4_lrd1e-6_linear --threshold_st 0.6 -k 100 --lambda_fm 0.1 --learning_rate 0.001 --learning_rate_D 0.000001 --lambda_adv 0.1 --in_channels_d 3 --weight_decay 0.0001
#python train_s4GAN.py --save ./work/gan/s4gan_lr1e-3_wd1e-4_lrd1e-6 --threshold_st 0.6 -k 100 --lambda_fm 0.1 --learning_rate 0.001 --learning_rate_D 0.000001 --lambda_adv 0.1 --in_channels_d 3 --weight_decay 0.0001
#python train_s4GAN.py --save ./work/gan/s4gan_lr1e-3_wd1e-4_lrd1e-5 --threshold_st 0.6 -k 100 --lambda_fm 0.1 --learning_rate 0.001 --learning_rate_D 0.00001 --lambda_adv 0.1 --in_channels_d 3 --weight_decay 0.0001
#python train_s4GAN.py --save ./work/gan/s4gan_lr1e-3_wd1e-4_lrd1e-4 --threshold_st 0.6 -k 100 --lambda_fm 0.1 --learning_rate 0.001 --learning_rate_D 0.0001 --lambda_adv 0.1 --in_channels_d 3 --weight_decay 0.0001

# before summer holiday
#python train_s4GAN.py --save ./work/gan/s4gan_lr_1e-4_wd5e-4_lrd1e-5_withgrad --threshold_st 0.6 -k 100 --lambda_fm 0.1 --learning_rate 0.0001 --learning_rate_D 0.00001 --lambda_adv 0.1 --in_channels_d 3 --weight_decay 0.0005
#python train_s4GAN.py --save ./work/gan/s4gan_thst0.6_lfm0.1_dorihardlabel_lr_1e-4_wd5e-4_lrd1e-5_k100_ladv0.1_withgrad --threshold_st 0.6 -k 100 --lambda_fm 0.1 --learning_rate 0.0001 --learning_rate_D 0.00001 --lambda_adv 0.1 --in_channels_d 3 --weight_decay 0.0005
#python train_s4GAN.py --save ./work/gan/s4gan_thst0.6_lfm0.1_dorilabel_lr_1e-4_wd5e-4_lrd1e-5_k100_ladv0.1 --threshold_st 0.6 -k 100 --lambda_fm 0.1 --learning_rate 0.0001 --learning_rate_D 0.00001 --lambda_adv 0.1 --in_channels_d 3 --weight_decay 0.0005
#python train_s4GAN.py --save ./work/gan/s4gan_thst0.6_lfm0.1_dorilabel_lr_1e-4_wd5e-4_lrd1e-5_k100_ladv0.1_withgrad --threshold_st 0.6 -k 100 --lambda_fm 0.1 --learning_rate 0.0001 --learning_rate_D 0.00001 --lambda_adv 0.1 --in_channels_d 3 --weight_decay 0.0005
#python train_s4GAN.py --save ./work/gan/s4gan_thst0.6_lfm0.1_dorilabel_lr_2.5e-4_wd5e-4_lrd2.5e-4_k100_ladv0.1_withgrad --threshold_st 0.6 -k 100 --lambda_fm 0.1 --learning_rate 0.00025 --learning_rate_D 0.00025 --lambda_adv 0.1 --in_channels_d 3 --weight_decay 0.0005

#python train_s4GAN.py --save ./work/gan/s4gan_thst0.6_lfm0.1_dorilabel_lr_0.5e-4_wd1e-4_lrd1e-6_k100_ladv0.1 --threshold_st 0.6 -k 100 --lambda_fm 0.1 --learning_rate_D 0.000001 --lambda_adv 0.1 --in_channels_d 3 --weight_decay 0.0001
#python train_s4GAN.py --save ./work/gan/s4gan_thst0.6_lfm0.1_dorilabel_lr_0.5e-4_lrd1e-6_k100_ladv0.1 --threshold_st 0.6 -k 100 --lambda_fm 0.1 --learning_rate_D 0.000001 --lambda_adv 0.1 --in_channels_d 3
#python train_s4GAN.py --save ./work/gan/s4gan_thst0.6_lfm0.1_dorilabel_lr_1e-5_lrd1e-6_k100_ladv0.1 --threshold_st 0.6 -k 100 --lambda_fm 0.1 --learning_rate_D 0.000001 --lambda_adv 0.1 --in_channels_d 3
#python train_s4GAN.py --save ./work/gan/s4gan_thst0.5_lfm0.1_dlabel_lrd1e-4_k100_ladv0.1_withgrad --threshold_st 0.6 -k 100 --lambda_fm 0.1 --learning_rate_D 0.0001 --lambda_adv 0.1 --in_channels_d 2
#python train_s4GAN.py --save ./work/gan/s4gan_thst0.5_lfm0.1_dlabel_lrd1e-6_k100_ladv0.1_withgrad --threshold_st 0.6 -k 100 --lambda_fm 0.1 --learning_rate_D 0.000001 --lambda_adv 0.1 --in_channels_d 2
#python train_s4GAN.py --save ./work/gan/s4gan_thst0.5_lfm0.1_dorilabel_lrd1e-6_k100_ladv0.1_withgrad --threshold_st 0.6 -k 100 --lambda_fm 0.1 --learning_rate_D 0.000001 --lambda_adv 0.1
#python train_s4GAN.py --save ./work/gan/s4gan_thst0.5_lfm0.1_dorilabel_lrd1e-6_k100_ladv0.01 --threshold_st 0.6 -k 100 --lambda_fm 0.1 --learning_rate_D 0.000001 --lambda_adv 0.01
#python train_s4GAN.py --save ./work/gan/s4gan_thst0.5_lfm0.1_dorilabel_lrd1e-6_k100_ladv1 --threshold_st 0.6 -k 100 --lambda_fm 0.1 --learning_rate_D 0.000001 --lambda_adv 1
#python train_s4GAN.py --save ./work/gan/s4gan_thst0.5_lfm0.1_dorilabel_lrd1e-6_adam_k100_lossadv1 --threshold_st 0.6 -k 100 --lambda_fm 0.1 --learning_rate_D 0.000001 --lambda_adv 1
#python train_s4GAN.py --save ./work/gan/s4gan_thst0.5_lfm0.1_dorilabel_lrd1e-6_adam_k100_lossadv --threshold_st 0.6 -k 100 --lambda_fm 0.1 --learning_rate_D 0.000001 --lambda_adv 0.1
#python train_s4GAN.py --save ./work/gan/s4gan_thst0.5_lfm0.1_dorilabel_lrd1e-6_k100 --threshold_st 0.6 -k 100 --lambda_fm 0.1 --learning_rate_D 0.000001
#python train_s4GAN.py --save ./work/gan/s4gan_thst0.5_lfm0.1_dorilabel_lrd1e-4_k100 --threshold_st 0.6 -k 100 --lambda_fm 0.1 --learning_rate_D 0.0001
#python train_s4GAN.py --save ./work/gan/s4gan_thst0.5_lfm0.1_dorilabel_k100 --threshold_st 0.6 -k 100 --lambda_fm 0.1
#python train_s4GAN.py --save ./work/gan/s4gan_thst0.5_lfm0.1_dlabel_k100 --threshold_st 0.6 -k 100 --lambda_fm 0.1
#python train_s4GAN.py --save ./work/gan/s4gan_thst0.5_lfm1_dlabel_k100 --threshold_st 0.5 -k 100 --lambda_fm 1
# ----


# train uate with new formular
#python ./train.py --save ./work/semi/uate_100_new_l30 --max_val 0.1 --train_method semisuper --is_uncertain --alpha_psudo 0.6 --sample_k 100 
#python ./train.py --save ./work/semi/uate_100_new_l30_1 --max_val 0.1 --train_method semisuper --is_uncertain --alpha_psudo 0.6 --sample_k 100 
#python ./train.py --save ./work/semi/uate_100_new_l30_2 --max_val 0.1 --train_method semisuper --is_uncertain --alpha_psudo 0.6 --sample_k 100 
#python ./train.py --save ./work/semi/uate_100_new_l30_3 --max_val 0.1 --train_method semisuper --is_uncertain --alpha_psudo 0.6 --sample_k 100 
#python ./train.py --save ./work/semi/uate_300_new_l30 --max_val 0.1 --train_method semisuper --is_uncertain --alpha_psudo 0.6 --sample_k 300 
#python ./train.py --save ./work/semi/uate_885_new_l30 --max_val 0.1 --train_method semisuper --is_uncertain --alpha_psudo 0.6 --sample_k 885 
#python ./train.py --save ./work/semi/uate_1770_new_l30 --max_val 0.1 --train_method semisuper --is_uncertain --alpha_psudo 0.6 --sample_k 1770 
#python ./train.py --save ./work/semi/uate_4428_new_l30 --max_val 0.1 --train_method semisuper --is_uncertain --alpha_psudo 0.6 --sample_k 4428 


#python ./train.py --save ./work/methods/uate_visual --max_val 0.1 --train_method semisuper --is_uncertain --sample_k 100 
#python ./train.py --save ./work/methods/uate_visual_1 --max_val 0.1 --train_method semisuper --is_uncertain --sample_k 100 


# 2020.08.13 semi-seg using 300 labeled images under diff methods
#python train_pi.py --log_dir ./log/methods --max_val 0.1 --save ./work/methods/pi_model_300 -k 300
#python train_tcsm.py --log_dir ./log/methods --save ./work/methods/tcsm_300 -k 300
#python ./train_mt.py --save ./work/methods/mean_teacher_300 --max_val 1 --sample_k 300 
#python ./train_mt.py --save ./work/methods/ua_mt_300 --max_val 0.1 --sample_k 300 --is_uncertain
#python train_gan.py --log_dir ./log/methods --save ./work/methods/gan_300 -k 300
#python ./train.py --save ./work/semi/te_300 --max_val 1 --train_method semisuper --sample_k 300
#python train_tcsm.py --log_dir ./log/methods --save ./work/methods/tcsm_300_1 -k 300
#python train_tcsm.py --log_dir ./log/methods --save ./work/methods/tcsm_300_2 -k 300 --max_val 0.01

# uate 300 diff T
#python ./train.py --save ./work/methods/uate_300_T5 --max_val 0.1 --train_method semisuper --is_uncertain --sample_k 300 -T 4
#python ./train.py --save ./work/methods/uate_300_T10 --max_val 0.1 --train_method semisuper --is_uncertain --sample_k 300 -T 9
#python ./train.py --save ./work/methods/uate_300_T2 --max_val 0.1 --train_method semisuper --is_uncertain --sample_k 300 -T 1


# semi-seg using 100 labeled images under diff methods
#python train_pi.py --log_dir ./log/methods --max_val 0.1 --save ./work/methods/pi_model
#python train_tcsm.py --log_dir ./log/methods --save ./work/methods/tcsm
#python train_gan.py --log_dir ./log/methods --save ./work/methods/gan
#python ./train_mt.py --save ./work/methods/mean_teacher --max_val 0.1 --sample_k 100 
#python ./train_mt.py --save ./work/methods/mean_teacher_w1 --max_val 1 --sample_k 100 
#python ./train_mt.py --save ./work/methods/ua_mt --max_val 0.1 --sample_k 100 --is_uncertain

# uate diff number of labeled images
#python ./train.py --save ./work/semi/te_100 --max_val 1 --train_method semisuper --sample_k 100 
#python ./train.py --save ./work/semi/uate_100 --max_val 0.1 --train_method semisuper --is_uncertain --sample_k 100 
#python ./train.py --save ./work/semi/uate_100_p10 --max_val 0.1 --train_method semisuper --is_uncertain --sample_k 100 
#python ./train.py --save ./work/semi/uate_100_p10_new --max_val 0.1 --train_method semisuper --is_uncertain --sample_k 100 
#python ./train.py --save ./work/semi/uate_100_uncer --max_val 0.1 --train_method semisuper --is_uncertain --sample_k 100 
#python ./train.py --save ./work/semi/uate_100_uncer --max_val 0.1 --train_method semisuper --is_uncertain --sample_k 100 
#python ./train.py --save ./work/semi/uate_100_Z --max_val 0.1 --train_method semisuper --is_uncertain --sample_k 100 
#python ./train.py --save ./work/semi/uate_100_Z_0.6 --max_val 0.1 --train_method semisuper --is_uncertain --alpha_psudo 0.6 --sample_k 100 
#python ./train.py --save ./work/semi/uate_300 --max_val 0.1 --train_method semisuper --is_uncertain --sample_k 300 
#python ./train.py --save ./work/semi/uate_885 --max_val 0.1 --train_method semisuper --is_uncertain --sample_k 885 
#python ./train.py --save ./work/semi/uate_1770 --max_val 0.1 --train_method semisuper --is_uncertain --sample_k 1770 
#python ./train.py --save ./work/semi/uate_4428 --max_val 0.1 --train_method semisuper --is_uncertain --sample_k 4428 
#python ./train.py --save ./work/semi/uate_8856 --max_val 0.1 --train_method semisuper --is_uncertain --sample_k 8856 
