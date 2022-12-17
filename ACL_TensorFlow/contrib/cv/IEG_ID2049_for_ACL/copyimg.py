import os
import shutil
filelist=os.listdir("/home/test_user01/slottl/tools-master/img2bin/imgout/")
for i in range(100):
    filepath=os.path.join("/home/test_user01/slottl/tools-master/img2bin/imgout/", filelist[i])
    shutil.copy(filepath,"./partimg")