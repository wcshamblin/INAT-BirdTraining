export TFHUB_CACHE_DIR='./tfhubcache/'
make_image_classifier \
  --image_dir /usr/xtmp/wcs/inaturalist/train_val2019/Birds \
  --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4 \
  --image_size 224 \
  --saved_model_dir /usr/xtmp/wcs/inat-models/inat-birds/ \
  --labels_output_file /usr/xtmp/wcs/inat-models/inat-birds/class_labels.txt \
  --tflite_output_file /usr/xtmp/wcs/inat-models/inat-birds/inat-birds.tflite \
  --log_dir /usr/xtmp/wcs/traininglogs/
