SrcPATH=/home/HwHiAiUser/Dev-Net/om
echo "[INFO] Src path: "$SrcPATH

PBModel=$SrcPATH/models/dev-net.pb
echo "[INFO] Pb Model file: "$PBModel

OMModel=$SrcPATH/models/dev-net
echo "[INFO] Om Model file: "$OMModel

atc --model=$PBModel --framework=3 --output=$OMModel --soc_version=Ascend310 --output_type=FP32
