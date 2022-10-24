atc --model=smith.pb --framework=3 --output=pb_res --soc_version=Ascend910 --input_shape="input_ids_1:32,2048;input_mask_1:32,2048;input_ids_2:32,2048;input_mask_2:32,2048" --out_nodes="seq_rep_from_bert_doc_dense/l2_normalize_1:0;Sigmoid:0;Round:0" --log=debug

