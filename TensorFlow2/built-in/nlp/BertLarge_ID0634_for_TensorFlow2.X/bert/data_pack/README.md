## Wikipedia pre-training data

Follow the Mlcommons' refereence implementation instructions to construct the training and eval datasets

## Pack sequences to reduce padding:

First convert the tfrecords to a binary format using 'bert_data/record_to_binary.py'
```
python3 bert_data/record_to_binary.py --tf-record-glob="path/to/your/unpacked/data/part*.tfrecord" --output-path="path/to/store/binary/files"
```
Then pack the sequence data using 'pack_pretraining_data':
```
python3 pack_pretarining_data.py --input-glob="path/to/store/binary/files" --output-dir="packed/data/folder"
```
The same stepe should also be repeated for the eval dataset.
The wikipedia dataset is now ready to be used in the Graphcore BERT model.
