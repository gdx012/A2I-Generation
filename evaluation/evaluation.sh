export IMAGE_DIR='/PATH/TO/YOUR/IMAGE/DIRECTORY'         
export AUDIO_DIR="/PATH/TO/YOUR/AUDIO/DIRECTORY"
# export DATASET_NAME='landscape'
export DATASET_NAME='A2I'

python inference.py --ref_batch VIRTUAL_imagenet256_labeled.npz --sample_batch $IMAGE_DIR
python aes.py -d $DATASET_NAME -i $IMAGE_DIR -a $AUDIO_DIR
python tis.py -d $DATASET_NAME --pred $IMAGE_DIR --gt $AUDIO_DIR
python ais.py -d $DATASET_NAME -i $IMAGE_DIR -a $AUDIO_DIR
python HPSv3.py -d $DATASET_NAME --pred $IMAGE_DIR --gt $AUDIO_DIR