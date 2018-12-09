mkdir -p ./data
mkdir -p ./training_data
gsutil -m cp "gs://quickdraw_dataset/full/numpy_bitmap/*" ./data
python create-dataset.py --npy_path data --output_path training_data