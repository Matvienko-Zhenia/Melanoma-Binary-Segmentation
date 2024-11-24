curl -X 'POST' \
  'http://127.0.0.1:8000/nn_inference' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@../IMD036.bmp' \
  -F "nn_type=unet" \
  -o segmentation.png
