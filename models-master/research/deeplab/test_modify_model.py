import six
import tensorflow as tf
from deeplab import common
from deeplab import model
from deeplab.datasets import segmentation_dataset
from deeplab.datasets import segmentation_dataset_only_person
from deeplab.utils import input_generator
from deeplab.utils import train_utils
from deployment import model_deploy
from tensorflow.python import pywrap_tensorflow

slim = tf.contrib.slim

prefetch_queue = slim.prefetch_queue

model_dir = '/home/lc/models-master/research/deeplab/datasets/pascal_voc_seg/init_models/deeplabv3_pascal_train_aug/model.ckpt.data-00000-of-00001'
exclude_list = ['global_step']
# if not initialize_last_layer:
# exclude_list.extend(last_layers)

variables_to_restore = slim.get_variables_to_restore(exclude=exclude_list)
reader = slim.assign_from_checkpoint_fn(model_dir, variables_to_restore, False)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key))
