#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from sys import argv
from bigdl.util.tf_utils import dump_model

import tensorflow as tf
import os.path as op
import os

def main():
    """
    How to run this script:
    python export_tf_checkpoint.py meta_file checkpoint_name save_path
    
    or
    
    python export_tf_checkpoint.py checkpoint_name save_path
    """
    meta_file = ""
    checkpoint = ""
    save_path = "model"

    if len(argv) == 2:
        meta_file = argv[1] + ".meta"
        checkpoint = argv[1]
    elif len(argv) == 3:
        meta_file = argv[1] + ".meta"
        checkpoint = argv[1]
        save_path = argv[2]
    elif len(argv) == 4:
        meta_file = argv[1]
        checkpoint = argv[2]
        save_path = argv[3]
    else:
        print("Invalid script arguments. How to run the script:\n" +
              "python export_tf_checkpoint.py checkpoint_name\n" +
              "python export_tf_checkpoint.py checkpoint_name save_path\n" +
              "python export_tf_checkpoint.py meta_file checkpoint_name save_path")
        exit(1)

    saver = tf.train.import_meta_graph(meta_file, clear_devices=True)

    if op.isfile(save_path):
        print("The save folder is a file. Exit")
        exit(1)

    if not op.exists(save_path):
        print("create folder")
        os.makedirs(save_path)

    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        dump_model(save_path, None, sess, checkpoint)

if __name__ == "__main__":
    main()