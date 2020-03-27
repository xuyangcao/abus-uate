python ./train.py --save ./work/semi/te_100 --max_val 1 --train_method semisuper --sample_k 100 
python ./train.py --save ./work/semi/uate_100 --max_val 0.1 --train_method semisuper --is_uncertain --sample_k 100 

python ./train.py --save ./work/semi/te_885 --max_val 1 --train_method semisuper --sample_k 885 
python ./train.py --save ./work/semi/uate_885 --max_val 0.1 --train_method semisuper --is_uncertain --sample_k 885 

python ./train.py --save ./work/semi/te_1770 --max_val 1 --train_method semisuper --sample_k 1770 
python ./train.py --save ./work/semi/uate_1770 --max_val 0.1 --train_method semisuper --is_uncertain --sample_k 1770 

python ./train.py --save ./work/semi/te_4428 --max_val 1 --train_method semisuper --sample_k 4428 
python ./train.py --save ./work/semi/uate_4428 --max_val 0.1 --train_method semisuper --is_uncertain --sample_k 4428 
