cur_path='pwd'
python3 ${cur_path}/train.py --epochs=5 --data_path=. --batch_size=4096 --sample_num=10000 --ckpt_save_path="" --precision_mode="" > loss+perf_gpu.txt 2>&1

