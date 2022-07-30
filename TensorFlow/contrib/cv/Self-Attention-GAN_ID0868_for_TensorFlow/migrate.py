import tensorflow as tf
import tensorflow.compat.v1 as tf1

def get_session_config(use_npu=False):
    if not use_npu:
        return tf1.ConfigProto()
    
    from npu_bridge.npu_init import RewriterConfig, npu_ops, npu_unary_ops, NPURunConfig, NPUEstimator
    from npu_bridge.estimator.npu import npu_convert_dropout

    tf.nn.gelu = tf1.nn.gelu = npu_unary_ops.gelu
    tf.nn.dropout = tf1.nn.dropout = npu_ops.dropout
    tf.estimator.RunConfig = tf1.estimator.RunConfig = NPURunConfig
    tf.estimator.Estimator = tf1.estimator.Estimator = NPUEstimator

    npu_config = tf1.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    custom_op = npu_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"

    # 在昇腾AI处理器执行训练，默认为True
    custom_op.parameter_map["use_off_line"].b = True

    # 开启混合计算模式，默认关闭
    # custom_op.parameter_map["mix_compile_mode"].b =  True
    # 混合计算场景下，指定in_out_pair中的算子是否下沉到昇腾AI处理器，取值：True：下沉，默认为True。False：不下沉。
    # custom_op.parameter_map['in_out_pair_flag'].b = False
    # 混合计算场景下，配置下沉/不下沉部分的首尾算子名。
    # all_graph_iop.append([in_nodes, out_nodes])
    # custom_op.parameter_map['in_out_pair'].s = tf.compat.as_bytes(str(all_graph_iop))

    # getnext算子是否下沉到昇腾AI处理器侧执行，默认为False，getnext算子下沉是迭代循环下沉的必要条件
    # custom_op.parameter_map["enable_data_pre_proc"].b = True

    # 训练迭代循环下沉，默认值为1表示不开启
    # 此处设置的值和set_iteration_per_loop设置的iterations_per_loop值保持一致，用于判断是否进行训练迭代下沉
    # custom_op.parameter_map["iterations_per_loop"].i = 10

    # 设置精度模式
    # 训练场景下，默认选择此种模式。当算子不支持float32数据类型时，直接降低精度到float16。当前不支持float32类型的算子都是卷积类算子，例如Conv2D、DepthwiseConv2D等，此类算子对精度不敏感，因此不会造成整网精度下降。
    # custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_fp32_to_fp16")
    # 在线推理场景下，默认选择此种模式。当算子既支持float16又支持float32数据类型时，强制选择float16。默认情况下系统使用此种方式，在线推理场景下，推荐使用该配置项。当算子既支持float16又支持float32数据类型时，强制选择float16。
    # custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp16")
    # 当算子既支持float16又支持float32数据类型时，强制选择float32。当算子既支持float16又支持float32数据类型时，强制选择float32。
    # custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp32")
    # 保持原图精度。此种方式下，如果整网中有Conv2D算子，由于该算子仅支持float16类型，在原图输入是float32类型的情况下，推理会报错中止。
    # custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("must_keep_origin_dtype")
    # 自动混合精度。可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。但需要注意的是：当前昇腾AI处理器仅支持float32到float16的精度调整。
    # custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    # 通过modify_mixlist指定需要修改的混合精度黑白灰算子名单
    # custom_op.parameter_map["modify_mixlist"].s = tf.compat.as_bytes("/home/test/ops_info.json")

    # 设置具体某个算子的精度模式，通过该参数传入自定义的精度模式配置文件op_precision.ini，可以为不同的算子设置不同的精度模式。
    # custom_op.parameter_map["op_precision_mode"].s = tf.compat.as_bytes("/home/test/op_precision.ini")

    # 是否开启变量格式优化。True：开启，默认开启。False：关闭。
    # custom_op.parameter_map["variable_format_optimize"].b =  True

    # 网络静态内存和最大动态内存，可根据网络大小指定。单位：Byte，取值范围：[0, 256*1024*1024*1024]或[0, 274877906944]。当前受昇腾AI处理器硬件限制，graph_memory_max_size和variable_memory_max_size总和最大支持31G。如果不设置，默认为26GB。
    # custom_op.parameter_map["graph_memory_max_size"].s = tf.compat.as_bytes(str(26*1024 * 1024 * 1024))

    # 变量内存，可根据网络大小指定。单位：Byte，取值范围：[0，256*1024*1024*1024]或[0, 274877906944]。当前受昇腾AI处理器硬件限制，graph_memory_max_size和variable_memory_max_size总和最大支持31G。如果不设置，默认为5GB。
    # custom_op.parameter_map["variable_memory_max_size"].s = tf.compat.as_bytes(str(5*1024 * 1024 * 1024))

    # 开启AutoTune自动调优模式，Auto Tune工具包含RL和GA两种调优模式
    # custom_op.parameter_map["auto_tune_mode"].s = tf.compat.as_bytes("RL,GA")

    # 指定AICPU/AICORE引擎的并行度，从而实现AICPU/AICORE算子间的并行执行。DNN_VM_TF为AICPU引擎名称，本示例指定了AICPU引擎的并发数为10；DNN_V100为AICORE引擎名称，本示例指定了AICORE引擎的并发数为1。AICPU/AICORE引擎的并行度默认为1，取值范围为：[1,13][1,3]。
    # custom_op.parameter_map["stream_max_parallel_num"].s = tf.compat.as_bytes("DNN_VM_TF:10,DNN_V100:1")

    # 分布式训练场景下，是否开启通信拖尾优化，用于提升训练性能。
    # custom_op.parameter_map["is_tailing_optimization"].b = True

    # 开启Profiling数据采集，当前支持采集的性能数据主要包括：
    # - training_trace：迭代轨迹数据，即训练任务及AI软件栈的软件信息，实现对训练任务的性能分析，重点关注数据增强、前后向计算、梯度聚合更新等相关数据。
    # - task_trace：任务轨迹数据，即昇腾AI处理器HWTS/AICore的硬件信息，分析任务开始、结束等信息。
    # - aicpu：采集aicpu数据增强的Profiling数据。
    # custom_op.parameter_map["profiling_mode"].b = True
    # custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/tmp/profiling","training_trace":"on","task_trace":"on","aicpu":"on","fp_point":"","bp_point":"","aic_metrics":"PipeUtilization","hccl":"on"}')

    # 开启Dump数据采集，当前支持采集的算子数据主要包括：
    # - input：dump算子的输入数据。
    # - output：dump算子的输出数据。
    # - all：同时dump算子的输入和输出数据。
    # enable_dump：是否开启Dump功能
    # custom_op.parameter_map["enable_dump"].b = True
    # dump_path：dump数据存放路径，该参数指定的目录需要在启动训练的环境上（容器或Host侧）提前创建且确保安装时配置的运行用户具有读写权限
    # custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes("/home/HwHiAiUser/output") 
    # dump_step：指定采集哪些迭代的Dump数据
    # custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes("0|5|10")
    # dump_mode：Dump模式，取值：input/output/all
    # custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all") 

    # 开启溢出数据采集，溢出检测目前有三种检测模式：
    # aicore_overflow：AI Core算子溢出检测模式，检测在算子输入数据正常的情况下，计算后的值是否为不正常的极大值（如float16下65500,38400,51200这些值）。一旦检测出这类问题，需要根据网络实际需求和算子逻辑来分析溢出原因并修改算子实现。
    # atomic_overflow：Atomic Add溢出检测模式，在AICore计算完，由UB搬运到OUT时，产生的Atomic Add溢出问题。
    # all：同时进行AI Core算子溢出检测和Atomic Add溢出检测。
    # dump_path：dump数据存放路径，该参数指定的目录需要在启动训练的环境上（容器或Host侧）提前创建且确保安装时配置的运行用户具有读写权限
    # custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes("/home/HwHiAiUser/output") 
    # enable_dump_debug：是否开启溢出检测功能
    # custom_op.parameter_map["enable_dump_debug"].b = True
    # dump_debug_mode：溢出检测模式，取值：all/aicore_overflow/atomic_overflow
    # custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all") 

    # 图执行模式，取值：
    # - 0：在线推理场景下，请配置为0。
    # - 1：训练场景下，请配置为1，默认为1。
    # custom_op.parameter_map["graph_run_mode"].i = 1

    # 算子debug功能开关，取值：
    # 0：不开启算子debug功能，默认为0。
    # 1：开启算子debug功能，在训练脚本执行目录下的kernel_meta文件夹中生成TBE指令映射文件（算子cce文件*.cce、python-cce映射文件*_loc.json、.o和.json文件），用于后续工具进行AICore Error问题定位。
    # 2：开启算子debug功能，在训练脚本执行目录下的kernel_meta文件夹中生成TBE指令映射文件（算子cce文件*.cce、python-cce映射文件*_loc.json、.o和.json文件），并关闭ccec编译器的编译优化开关且打开ccec调试功能（ccec编译器选项设置为-O0-g），用于后续工具进行AICore Error问题定位。
    # 3：不开启算子debug功能，且在训练脚本执行目录下的kernel_meta文件夹中保留.o和.json文件。
    # 4：不开启算子debug功能，在训练脚本执行目录下的kernel_meta文件夹中保留.o（算子二进制文件）和.json文件（算子描述文件），生成TBE指令映射文件（算子cce文件*.cce）和UB融合计算描述文件（{$kernel_name}_compute.json）。
    # custom_op.parameter_map["op_debug_level"].i = 0

    # 指定编译时需要生效的Scope融合规则列表。此处传入注册的融合规则名称，允许传入多个，用“,”隔开。
    # custom_op.parameter_map["enable_scope_fusion_passes"].s = tf.compat.as_bytes("ScopeLayerNormPass,ScopeClipBoxesPass")

    # 是否dump异常算子的输入和输出数据。
    # - 0：关闭，默认为0。
    # - 1：开启，用户可常开，不影响性能。
    # custom_op.parameter_map["enable_exception_dump"].i = 1

    # 昇腾AI处理器部分内置算子有高精度和高性能实现方式，用户可以通过该参数配置模型编译时选择哪种算子。取值包括：
    # high_precision：表示算子选择高精度实现。高精度实现算子是指在fp16输入的情况下，通过泰勒展开/牛顿迭代等手段进一步提升算子的精度。
    # high_performance：表示算子选择高性能实现。高性能实现算子是指在fp16输入的情况下，不影响网络精度前提的最优性能实现。默认为high_performance
    # custom_op.parameter_map["op_select_implmode"].s = tf.compat.as_bytes("high_precision")

    # 列举算子optype的列表，该列表中的算子使用op_select_implmode参数指定的模式，当前支持的算子为Pooling、SoftmaxV2、LRN、ROIAlign，多个算子以英文逗号分隔。
    # 该参数需要与op_select_implmode参数配合使用，例如：
    # op_select_implmode配置为high_precision。
    # optypelist_for_implmode配置为Pooling。
    # custom_op.parameter_map["optypelist_for_implmode"].s = tf.compat.as_bytes("Pooling,SoftmaxV2")

    # 输入的shape信息。
    # custom_op.parameter_map["input_shape"].s = tf.compat.as_bytes("data:1,1,40,-1;label:1,-1;mask:-1,-1")
    # 输入的对应维度的档位信息。
    # custom_op.parameter_map["dynamic_dims"].s = tf.compat.as_bytes("20,20,1,1;40,40,2,2;80,60,4,4")
    # 指定动态输入的节点类型。0：dataset输入为动态输入。1：placeholder输入为动态输入。当前不支持dataset和placeholder输入同时为动态输入。
    # custom_op.parameter_map["dynamic_node_type"].i = 0

    # 是否开启buffer优化。l2_optimize：表示开启buffer优化，默认为l2_optimize。off_optimize：表示关闭buffer优化。
    # custom_op.parameter_map["buffer_optimize"].s = tf.compat.as_bytes("l2_optimize")

    # 是否使能small channel的优化，使能后在channel<=4的卷积层会有性能收益。建议用户在推理场景下打开此开关。0：关闭，默认为0。1：使能。
    # custom_op.parameter_map["enable_small_channel"].i = 1

    # 融合开关配置文件路径以及文件名。
    # custom_op.parameter_map["fusion_switch_file"].s = tf.compat.as_bytes("/home/test/fusion_switch.cfg")

    # 用于配置算子编译磁盘缓存模式。
    # custom_op.parameter_map["op_compiler_cache_mode"].s = tf.compat.as_bytes("enable")
    # 用于配置算子编译磁盘缓存的目录。
    # custom_op.parameter_map["op_compiler_cache_dir"].s = tf.compat.as_bytes("/home/test/kernel_cache")

    # 用于配置保存算子编译生成的调试相关的过程文件的路径，包括算子.o/.json/.cce等文件。默认生成在当前脚本执行路径下。
    # custom_op.parameter_map["debug_dir"].s = tf.compat.as_bytes("/home/test")

    # 当前网络的输入是否为动态输入，取值包括：
    # - True：动态输入。
    # - False：固定输入，默认False。
    # custom_op.parameter_map["dynamic_input"].b = True
    # 对于动态输入场景，需要通过该参数设置执行模式。
    # dynamic_execute：动态图编译模式。该模式下获取dynamic_inputs_shape_range中配置的shape范围进行编译。
    # custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("dynamic_execute")
    # lazy_recompile
    # custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
    # 设置的shape范围
    # custom_op.parameter_map["dynamic_inputs_shape_range"].s = tf.compat.as_bytes("getnext:[128 ,3~5, 2~128, -1],[64 ,3~5, 2~128, -1];data:[128 ,3~5, 2~128, -1]")

    # 该参数用于推荐网络场景的数据并行场景，在主进程中对于数据进行去重操作，去重之后的数据再分发给其他进程的Device进行前后向计算。
    # custom_op.parameter_map["local_rank_id"].i = 0
    # 该参数配合local_rank_id使用，用来指定主进程给哪些其他进程的Device发送数据。
    # custom_op.parameter_map["local_device_list"].s = tf.compat.as_bytes("0,1")

    # 当用户需要将不同的模型通过同一个脚本在不同的Device上执行，可以通过该参数指定Device的逻辑ID。
    # 通常可以为不同的图创建不同的Session，并且传入不同的session_device_id，该参数优先级高于ASCEND_DEVICE_ID。
    # custom_op.parameter_map["session_device_id"].i = 0

    # 允许在没有昇腾AI处理器直接连接的服务器上启动训练或推理进程，通过分布式部署能力进行计算任务的远程部署。在该场景下，需要用户指定远程昇腾AI处理器版本号，用于编译优化对应计算任务。
    # custom_op.parameter_map["soc_config"].s = tf.compat.as_bytes("Ascend910")
    # 集合通信超时时间，单位为s。
    # custom_op.parameter_map["hccl_timeout"].i = 600
    # 算子等待超时时间，单位为s。
    # custom_op.parameter_map["op_wait_timeout"].i = 120
    # 算子执行超时时间，单位为s。
    # custom_op.parameter_map["op_execute_timeout"].i = 90

    # 以下配置默认关闭，请勿开启：
    # npu_config.graph_options.rewrite_options.disable_model_pruning = RewriterConfig.OFF

    # 以下配置默认开启，请勿关闭：
    # npu_config.graph_options.rewrite_options.function_optimization = RewriterConfig.ON
    # npu_config.graph_options.rewrite_options.constant_folding = RewriterConfig.ON
    # npu_config.graph_options.rewrite_options.shape_optimization = RewriterConfig.ON
    # npu_config.graph_options.rewrite_options.arithmetic_optimization = RewriterConfig.ON
    # npu_config.graph_options.rewrite_options.loop_optimization = RewriterConfig.ON
    # npu_config.graph_options.rewrite_options.dependency_optimization = RewriterConfig.ON
    # npu_config.graph_options.rewrite_options.layout_optimizer = RewriterConfig.ON

    # 以下配置默认开启，必须显式关闭：
    npu_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    npu_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

    tf1.logging.info('Graph Options: %s', npu_config.graph_options)

    return npu_config