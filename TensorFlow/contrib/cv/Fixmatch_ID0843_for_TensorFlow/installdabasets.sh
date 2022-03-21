export ML_DATA="path to where you want the datasets saved"
export PYTHONPATH=$PYTHONPATH:"path to the FixMatch"

# Download datasets
./scripts/create_datasets.py
cp $ML_DATA/svhn-test.tfrecord $ML_DATA/svhn_noextra-test.tfrecord

# Create unlabeled datasets
scripts/create_unlabeled.py $ML_DATA/SSL2/svhn $ML_DATA/svhn-train.tfrecord $ML_DATA/svhn-extra.tfrecord &
scripts/create_unlabeled.py $ML_DATA/SSL2/svhn_noextra $ML_DATA/svhn-train.tfrecord &
scripts/create_unlabeled.py $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord &
scripts/create_unlabeled.py $ML_DATA/SSL2/cifar100 $ML_DATA/cifar100-train.tfrecord &
scripts/create_unlabeled.py $ML_DATA/SSL2/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord &
wait

# Create semi-supervised subsets
for seed in 0 1 2 3 4 5; do
    for size in 10 20 30 40 100 250 1000 4000; do
        scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/svhn $ML_DATA/svhn-train.tfrecord $ML_DATA/svhn-extra.tfrecord &
        scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/svhn_noextra $ML_DATA/svhn-train.tfrecord &
        scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/cifar10 $ML_DATA/cifar10-train.tfrecord &
    done
    for size in 400 1000 2500 10000; do
        scripts/create_split.py --seed=$seed --size=$size $ML_DATA/SSL2/cifar100 $ML_DATA/cifar100-train.tfrecord &
    done
    scripts/create_split.py --seed=$seed --size=1000 $ML_DATA/SSL2/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord &
    wait
done
scripts/create_split.py --seed=1 --size=5000 $ML_DATA/SSL2/stl10 $ML_DATA/stl10-train.tfrecord $ML_DATA/stl10-unlabeled.tfrecord
