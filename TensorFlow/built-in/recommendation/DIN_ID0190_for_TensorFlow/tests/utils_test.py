from npu_bridge.npu_init import *
from deepctr.utils import check_version


def test_check_version():
    check_version('0.1.0')
    check_version(20191231)

