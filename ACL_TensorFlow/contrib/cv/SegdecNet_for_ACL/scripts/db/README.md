The full dataset Kolektor Surface Defect Dataset (KolektorSDD) is available here.
We split the dataset into three folds to perform 3-fold cross validation. The splits are available at http://box.vicos.si/skokec/gostop/KolektorSDD-training-splits.zip.
Fully prepared TensorFlow dataset split into 3 folds is available at http://box.vicos.si/skokec/gostop/KolektorSDD-dilate=5-tensorflow.zip.
The directory level here should be：
db|KolektorSDD-dilate=5|fold_0|test-00000-of-00001
                              |test_ids.txt
                              |train-00000-of-00001
                              |train_ids.txt
                       |fold_1|test-00000-of-00001
                              |test_ids.txt
                              |train-00000-of-00001
                              |train_ids.txt
                       |fold_2|test-00000-of-00001
                              |test_ids.txt
                              |train-00000-of-00001
                              |train_ids.txt