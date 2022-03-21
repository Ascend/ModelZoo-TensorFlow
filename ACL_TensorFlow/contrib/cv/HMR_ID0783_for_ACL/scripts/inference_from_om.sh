# TODO: Replace with path to the om model you want to evaluate.
OM=om_model.om
# TODO: Replace with where you saved the converted test data.
CONVERTED=om_test_data
# TODO: Replace with where you want to save the output.
OUTPUT=om_output


msame --model ${OM} --input ${CONVERTED}/input --output ${OUTPUT}