# uate gan
python ./train_uategan.py --save ./work/uategan/uategan_100_Dunet_leakrelu_dropout --max_val 0.1 --sample_k 100 --is_uncertain --alpha_psudo 0.6
#python ./train_uategan.py --save ./work/uategan/uategan_100_Dunet --max_val 0.1 --sample_k 100 --is_uncertain --alpha_psudo 0.6
#python ./train_uategan.py --save ./work/uategan/uategan_bce_100_wd4 --max_val 0.1 --sample_k 100 --is_uncertain --alpha_psudo 0.6
#python ./train_uategan.py --save ./work/uategan/uategan_bce_100 --max_val 0.1 --sample_k 100 --is_uncertain --alpha_psudo 0.6
#python ./train_uategan.py --save ./work/uategan/uategan_baseline --max_val 0.1 --sample_k 100 --is_uncertain --alpha_psudo 0.6


# tegan_method
#python ./train_tegan.py --save ./work/tegan/tegan_300 --max_val 1 --sample_k 300 # reduce HD compread with te model
#python ./train_tegan.py --save ./work/tegan/tegan_100 --max_val 1 --sample_k 100 # reduce HD compread with te model

# emt_vat 
#python ./train_emt_vat.py --save ./work/gan_task2/emt_885_ori --max_val 1 --sample_k 885 
#python ./train_emt_vat.py --save ./work/gan_task2/emt_300_ori --max_val 1 --sample_k 300 
#python ./train_emt_vat.py --save ./work/gan_task2/emt_100_ori --max_val 1 --sample_k 100 
#python ./train_emt_vat.py --save ./work/gan_task2/emt_vat1_100 --max_val 1 --sample_k 100 --is_vat
#python ./train_emt_vat.py --save ./work/gan_task2/emt_vat_mix10_100 --max_val 1 --sample_k 100 --mix
#python ./train_emt_vat.py --save ./work/gan_task2/emt_vat_mix_100 --max_val 1 --sample_k 100 --mix
#python ./train_emt_vat.py --save ./work/gan_task2/emt_vat_100 --max_val 1 --sample_k 100

#python ./train_lstm_te.py --save ./work/gan_task2/lstm_te_100_1 --max_val 1 --sample_k 100 
#python ./train_lstm_te.py --save ./work/gan_task2/lstm_te_100 --max_val 1 --sample_k 100 


#python ./train_mt_AC.py --save ./work/gan_task2/mtac_100 --max_val 1 --sample_k 100
#python ./train_tegan_ac.py --save ./work/gan_task2/tegan_ac_100_1gpu --max_val 1 --alpha_psudo 0.6 --sample_k 100 


# semi with artificial images
#python ./train_artifial.py --save ./work/gan_task2/artificial_labels_100 --max_val 1 --sample_k 100


# tegan cutmix
#python ./train_mask_tegan.py --save ./work/gan_task2/tegan_cutmix_100 --max_val 1 --sample_k 100


# cutmix methods_2
#python ./train_cutmix.py --save ./work/gan_task2/cutmix_mt_1770_1gpu --max_val 1 --sample_k 1770
#python ./train_cutmix.py --save ./work/gan_task2/cutmix_mt_885_1gpu --max_val 1 --sample_k 885
#python ./train_cutmix.py --save ./work/gan_task2/cutmix_mt_300_1gpu --max_val 1 --sample_k 300
#python ./train_cutmix.py --save ./work/gan_task2/cutmix_mt_100_1gpu --max_val 1
#python ./train_cutmix.py --save ./work/gan_task2/cutmix_mt_300_maxval5 --max_val 5 --sample_k 300
#python ./train_cutmix.py --save ./work/methods_2/cutmix_mt_300 --max_val 1 --sample_k 300
#python ./train_cutmix.py --save ./work/methods_2/cutmix_mt_100 --max_val 1

# mean teacher methods_2
#python ./train_mt.py --save ./work/methods_2/mean_teacher --max_val 1 --sample_k 100 


# uate methods_2
#python ./train.py --save ./work/methods_2/uate_100 --max_val 0.1 --train_method semisuper --is_uncertain --sample_k 100 

# te methods 2
#python ./train.py --save ./work/methods_2/te_1770 --max_val 1 --train_method semisuper --sample_k 1770 
#python ./train.py --save ./work/methods_2/te_885 --max_val 1 --train_method semisuper --sample_k 885 
#python ./train.py --save ./work/methods_2/te_300 --max_val 1 --train_method semisuper --sample_k 300 
#python ./train.py --save ./work/methods_2/te_100 --max_val 1 --train_method semisuper --sample_k 100 


# mtgan cutmix
#python ./train_mask_mtgan.py --save ./work/gan_task2/mtgan_cutmix_300_1gpu_rmadvloss --max_val 1 --sample_k 300
#python ./train_mask_mtgan.py --save ./work/gan_task2/mtgan_cutmix_100_1gpu_rmadvloss --max_val 1 --sample_k 100
#python ./train_mask_mtgan.py --save ./work/gan_task2/mtgan_cutmix_100_1gpu --max_val 1 --sample_k 100
#python ./train_mask_mtgan.py --save ./work/gan_task2/mtgan_cutmix_100 --max_val 1 --sample_k 100


#python ./train_tegan.py --save ./work/gan_task2/tegan_300_1gpu --max_val 1 --alpha_psudo 0.6 --sample_k 300 
#python ./train_tegan.py --save ./work/gan_task2/tegan_100_hard --max_val 1 --alpha_psudo 0.6 --sample_k 100 --consis_method hard
#python ./train_vatmt.py --save ./work/gan_task2/vatmt_100_singlegpu --max_val 1 --sample_k 100 
#python ./train_tegan.py --save ./work/gan_task2/tegan_100_singlegpu --max_val 1 --alpha_psudo 0.6 --sample_k 100 
#python ./train_tegan.py --save ./work/gan_task2/uategan_100_val0.1 --max_val 0.1 --alpha_psudo 0.6 --sample_k 100 --is_uncertain
#python ./train_tegan.py --save ./work/gan_task2/tegan_100_lrD1e-5 --max_val 0.1 --alpha_psudo 0.6 --sample_k 100 --lr_D 0.00001
#python ./train_tegan.py --save ./work/gan_task2/uategan_100 --max_val 1 --alpha_psudo 0.6 --sample_k 100 --is_uncertain
#python ./train_tegan.py --save ./work/gan_task2/tegan_300 --max_val 1 --alpha_psudo 0.6 --sample_k 300 
#python ./train_tegan.py --save ./work/gan_task2/tegan_100 --max_val 1 --alpha_psudo 0.6 --sample_k 100 

# mtgan method
#python train_mtgan.py --save ./work/gan_task2/mtgan_100_1gpu --max_val 1
#python train_mtgan.py --save ./work/gan_task2/mtgan_100 --max_val 1
#python train_mtgan.py --save ./work/gan_task2/mtgan_cons1_st_th0.55 --consistency 1 --threshold_st 0.55
#python train_mtgan.py --save ./work/gan_task2/mtgan_cons1_st --consistency 1 --threshold_st 0.5
#python train_mtgan.py --save ./work/gan_task2/mtgan_lr1e-3_lrd1e-4_cons0.1_st_1gpu --consistency 0.1 --lr 0.001 --lr_D 0.0001 --threshold_st 0.5
#python train_mtgan.py --save ./work/gan_task2/mtgan_lr3e-5_lrd1e-4_cons0.1_st_1gpu --consistency 0.1 --lr_D 0.0001 --threshold_st 0.5
#python train_mtgan.py --save ./work/gan_task2/mtgan_lr3e-5_lrd1e-4_cons0.1_st --consistency 0.1 --lr_D 0.0001 --threshold_st 0.5
#python train_mtgan.py --save ./work/gan_task2/mtgan_lr3e-5_lrd1e-4_cons0.1_visual --consistency 0.1 --lr_D 0.0001
#python train_mtgan.py --save ./work/gan_task2/mtgan_lr3e-5_lrd1e-4_cons0.1 --consistency 0.1 --lr_D 0.0001
#python train_mtgan.py --save ./work/gan_task2/mtgan_lr3e-5_lrd1e-6_cons0.1 --consistency 0.1
#python train_mtgan.py --save ./work/gan_task2/mtgan_lr3e-5_lrd1e-5_cons0.1_lfm --lr 0.00003 --lr_D 0.00001 --consistency 0.1
#python train_mtgan.py --save ./work/gan_task2/mtgan_lr3e-5_lrd1e-5_cons0.1 --lr 0.00003 --lr_D 0.00001 --consistency 0.1


#python train_s4GAN.py --save ./work/methods_2/s4gan_100
#python train_s4GAN.py --save ./work/gan_task2/s4gan_baseline


# finish s4GAN baseline 2020.08.28
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

#python ./train.py --save ./work/gan_task2/te_100_1gpu --max_val 1 --train_method semisuper --sample_k 100 

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
