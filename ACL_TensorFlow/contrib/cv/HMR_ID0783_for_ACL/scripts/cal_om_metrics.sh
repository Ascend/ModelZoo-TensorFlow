# TODO: Replace with path to load pred data.
PRED=om_output/20210928_164811
# TODO: Replace with path to load gt data.
GT=om_test_data/test_label.npy

CMD="python3 -u -m src.cal_om_metrics \
--pred_dir=${PRED} \
--gt_path=${GT}"

echo $CMD
$CMD