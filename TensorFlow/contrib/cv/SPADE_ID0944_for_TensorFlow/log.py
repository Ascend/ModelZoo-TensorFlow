import sys, logging

fmt = logging.Formatter(fmt="[%(levelname)s %(asctime)s.%(msecs)03d %(process)d %(filename)s:%(lineno)d] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

def set_logger():
    # 设置root logger日志格式
    root_logger = logging.getLogger()
    if len(root_logger.handlers):
        root_handler = root_logger.handlers[0]
        root_handler.setFormatter(fmt)

    # tensorflow专用logger，tf.logging使用
    tf_logger = logging.getLogger('tensorflow')
    # 不传播到root logger
    tf_logger.propagate = False
    # 设置 INFO(20), WARNING(30), ERROR(40), CRITICAL(50) 可见，DEBUG(10) 不可见
    tf_logger.setLevel(logging.INFO)
    # 删除默认handler
    if len(tf_logger.handlers):
        tf_logger.removeHandler(tf_logger.handlers[0])
    # 创建两个handler
    handler_stdout = logging.StreamHandler(sys.stdout)
    handler_stderr = logging.StreamHandler(sys.stderr)
    # 设置日志格式
    handler_stdout.setFormatter(fmt)
    handler_stderr.setFormatter(fmt)
    # 设置两个handler处理不同等级的日志
    class StdoutFilter(logging.Filter):
        def filter(self, record):
            return record.levelno < logging.WARNING
    handler_stdout.addFilter(StdoutFilter())
    handler_stderr.setLevel(logging.WARNING)
    # 添加handler
    tf_logger.addHandler(handler_stdout)
    tf_logger.addHandler(handler_stderr)