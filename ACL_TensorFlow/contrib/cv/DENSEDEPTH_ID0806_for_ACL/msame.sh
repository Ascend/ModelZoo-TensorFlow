#!/bin/bash
msame --model ./om/test.om --input ./result/bin/image --output ./result/bin_infer/image --outfmt BIN
msame --model ./om/test.om --input ./result/bin/image_flip --output ./result/bin_infer/image_flip --outfmt BIN