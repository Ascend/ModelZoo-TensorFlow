#!/bin/bash
msame --model=om/n2n_gaussian_h512_w768_910A.om --input=bin/kodak --output=result/bin/kodak --outfmt=BIN
msame --model=om/mri.om --input=bin/ixi_valid --output=result/bin/ixi_valid --outfmt=BIN
