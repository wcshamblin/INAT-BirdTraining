PIPELINE_CONFIG_PATH=/usr/xtmp/wcs/models/faster_rcnn_resnet101_fgvc_2018_07_19/pipeline.config
MODEL_DIR=/usr/xtmp/wcs/models/inat2017/
NUM_TRAIN_STEPS=10000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python3 object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
