# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import numpy as np

# Definition of arguments used in functions defined within this file

parser = argparse.ArgumentParser()

parser.add_argument('--filename', action='store', type=str, required=True, help='Include string ALPHA_UNDERSCORE where you want to modify each filename')

args = parser.parse_args()

if __name__ == "__main__":

    fn = args.filename
    f  = open('template_resnet.pbs', 'r')
    l  = f.read()
    
    # Generate rate-testing-pbs files based on template
    for i in np.arange(0.2, 3.1, 0.2):
    	digit1  = int(i)
    	digit01 = int(i*10-int(i)*10)
    	decim   = str(i)
    	und     = str(digit1)+"_"+str(digit01)
    
    	curr_fn  = fn.replace('ALPHA_UNDERSCORE', und)
    	curr_dat = l.replace('ALPHA_UNDERSCORE', und)
    	curr_dat = curr_dat.replace('ALPHA_DECIMAL', decim)
    
    	saveF = open(curr_fn, 'w')
    	saveF.write(curr_dat)
    	saveF.close()
    
    # END FOR
