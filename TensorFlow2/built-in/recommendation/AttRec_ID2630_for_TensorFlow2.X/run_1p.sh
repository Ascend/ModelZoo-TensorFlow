cur_path='pwd'
python3 ${cur_path}/train.py --epochs=40 --data_path=. --batch_size=1024 --ckpt_save_path="" --precision_mode="" > loss+perf_gpu.txt 2>&1

