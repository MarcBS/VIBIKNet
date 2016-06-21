export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export PYTHONPATH=/usr/local/C3D/python
export CPLUS_INCLUDE_PATH=/usr/include/python2.7

GLOG_logtosterr=1 %caffe_path%/build/tools/extract_image_features.bin %path_structure% %path_weights% %gpu% %batch_size% %num_batches% %list_images% %layer_name%
