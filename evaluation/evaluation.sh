export IMAGE_DIR='results/output_lr_5e5_FiLM/epoch_70_landscape/Fulldataset'         
export AUDIO_DIR='data/A2I_Set/eval_audio_all'
export DATASET_NAME='A2I'
# export AUDIO_DIR='data/landscape/test'
# export DATASET_NAME='landscape'


python evaluation/inference.py --ref_batch ./evaluation/VIRTUAL_imagenet256_labeled.npz --sample_batch $IMAGE_DIR
python evaluation/aes.py -d $DATASET_NAME -i $IMAGE_DIR -a $AUDIO_DIR
python evaluation/tis.py -d $DATASET_NAME --pred $IMAGE_DIR --gt $AUDIO_DIR
python evaluation/ais.py -d $DATASET_NAME -i $IMAGE_DIR -a $AUDIO_DIR
python evaluation/HPSv3.py -d $DATASET_NAME --pred $IMAGE_DIR --gt $AUDIO_DIR