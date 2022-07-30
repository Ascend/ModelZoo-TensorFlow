./msame --model munit.om \
	   --input image_test/valA,image_test/valB,image_test/style \
	   --output inference \
	   --outfmt BIN \
	   | tee msame.log