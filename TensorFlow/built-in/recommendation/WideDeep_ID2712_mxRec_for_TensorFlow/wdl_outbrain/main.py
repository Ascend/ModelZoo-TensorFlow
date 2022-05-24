# coding: UTF-8
import os

from mxRec.interface.dataset import Dataset
from mxRec.interface.table import TableEmbedding
from mxRec.interface.hook.callable import QPSHook, EvalHook
from mxRec.interface.model import Model
from mxRec.interface.optimizer import Adam, Ftrl
from mxRec.interface.session import Session
from mxRec.interface.task import Task
from example.wdl_outbrain.model import my_model
from example.wdl_outbrain.preprocess import my_preprocess
from sklearn.metrics import roc_auc_score


def run():
    #  ** config **
    batch_size = 131072  # note: eval bs 和 train bs 必须一致
    train_size = 59761827
    test_size = 1048576

    data_path = '/data/local/outbrain/tf_record'

    dataset_type = "tf_dataset"

    output_op_name = "logits"
    wide_loss_op_name = "wide_loss"
    deep_loss_op_name = "deep_loss"
    metrics_op_name = "metrics"
    pred_op_name = "pred"
    label_op_name = "label"

    print("Init a dataset instance.")
    data_args_json_file = os.path.join(data_path, 'data_args.json')
    train_dataset = Dataset(data_path, dataset_type,
                            name="train_dataset_channel", dataset_pattern="train",
                            batch_size=batch_size) \
        .read_parsing_json(data_args_json_file) \
        .file_pattern('tfrecord') \
        .set_num_parallel(8) \
        .shuffle(shuffle_buffer=1) \
        .prefetch(1) \
        .epoch(10) \
        .sample_size(train_size) \
        .add_preprocess(function=my_preprocess)

    val_dataset = Dataset(data_path, dataset_type,
                          name="test_dataset_channel", dataset_pattern="eval",
                          batch_size=batch_size) \
        .read_parsing_json(data_args_json_file) \
        .file_pattern('tfrecord') \
        .set_num_parallel(8) \
        .shuffle(shuffle_buffer=1) \
        .prefetch(1) \
        .sample_size(test_size) \
        .add_preprocess(function=my_preprocess)

    dirname = os.path.dirname(__file__)
    embedding = TableEmbedding(
        name="table_contents").set_embedding_domain_by_json(
        lookup_spec_file=os.path.join(
            dirname,
            "./lookup_spec_dct.json"),
        table_spec_file=os.path.join(
            dirname,
            "./table_spec_dct.json"))

    var_list_deep = "deep_dense"
    var_list_wide = "wide_dense"

    model = Model(function=my_model.build_model)\
        .add_output(output_op_name)\
        .add_loss([wide_loss_op_name, deep_loss_op_name])\
        .add_prediction(pred_op_name)\
        .add_label(label_op_name)\
        .add_metrics(metrics_op_name)\
        .add_var_list([var_list_deep, var_list_wide])

    adam = Adam(
        lr=1e-4,
        beta1=0.9,
        beta2=0.999,
        epsilon=9e-8,
        is_original=True).bind(
        deep_loss_op_name,
        var_list_deep)
    # Todo get_ops是否需要隐掉
    adam_op = adam.get_ops()
    npu_adam = Adam(
        lr=1e-4,
        beta1_power=1.0,  # note: 跑 psworker 模式必须配置beta1_power和beta2_power的值，待修复
        beta2_power=1.0,
        beta1=0.9,
        beta2=0.999,
        epsilon=9e-8).bind(deep_loss_op_name,
                           embedding.get_opt_var_list(Adam.opt_type))
    npu_adam_op = npu_adam.get_ops()

    ftrl = Ftrl(
        init_accum=1,
        learning_rate=0.06,
        l1=3e-8,
        l2=1e-6,
        l1_power=-0.5,
        is_original=True).bind(
        wide_loss_op_name,
        var_list_wide)
    ftrl_op = ftrl.get_ops()
    npu_ftrl = Ftrl(
        init_accum=1,
        learning_rate=0.06,
        l1=3e-8,
        l2=1e-6,
        l1_power=-
        0.5).bind(
        wide_loss_op_name,
        embedding.get_opt_var_list(Ftrl.opt_type))
    npu_ftrl_op = npu_ftrl.get_ops()

    # Todo get_op以及set module等操作应该也可以接受实例的传参

    training_task = Task("training", running_mode="training")\
        .set_dataset(train_dataset)\
        .set_model(model)\
        .add_opt(adam)\
        .add_opt(npu_adam)\
        .add_opt(ftrl)\
        .add_opt(npu_ftrl)

    eval_task = Task("eval", running_mode="eval")\
        .set_dataset(val_dataset)\
        .set_model(model)

    # Todo 以后再完善，目前尚未实现
    qps_hook = QPSHook(sess_run_only=False)
    pred_and_label_op_name = {"pred": pred_op_name, "label": label_op_name}
    eval_hook = EvalHook(
        pred_and_label_op_name,
        roc_auc_score,
        eval_task)

    with Session(hooks=[qps_hook, eval_hook]) as sess:
        while not sess.is_terminated():
            ret = sess.run([deep_loss_op_name, wide_loss_op_name,
                            adam_op, npu_adam_op, ftrl_op, npu_ftrl_op])

            print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>deep_loss: {ret[0]}")
            print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>wide_loss: {ret[1]}")
    print("Main func completed!")


if __name__ == "__main__":
    run()
    print("Done !")
