SrcPATH=/home/HwHiAiUser/Dev-Net/om
echo "[INFO] Src path: "$SrcPATH

Model=$SrcPATH/models/dev-net.om
echo "[INFO] Model file: "$Model

Input=$SrcPATH/data/test_input.bin
echo "[INFO] Input file: "$Input

Output=$SrcPATH/om_output
echo "[INFO] Output dir: "$Output

Msame=$SrcPATH/msame
echo "[INFO] Msame dir: "$Msame

$Msame --model $Model --input $Input --output $Output
