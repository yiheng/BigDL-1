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
from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.platform import gfile

def merge_checkpoint(input_graph,
                 input_checkpoint,
                 output_node_names,
                 output_graph):
    """
    merge the checkpoint file with the non-binary graph file to
    generate one GraphDef file with the variable values
    Args:
        input_graph: the GraphDef file, not in the binary form
        input_checkpoint: the checkpoint file
        output_node_names: A string of name of the output names, 
            use comma to seperate multi outputs
        output_graph: String of the location and the name of the
            output graph
    """
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"

    input_graph_def = graph_pb2.GraphDef()
    mode = "r"
    with gfile.FastGFile(input_graph, mode) as f:
        text_format.Merge(f.read().decode("utf-8"), input_graph_def)
    for node in input_graph_def.node:
      node.device = ""
    _ = importer.import_graph_def(input_graph_def, name="")
    with session.Session() as sess:
        sess.run([restore_op_name], {filename_tensor_name: input_checkpoint})
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(","),
            variable_names_blacklist="")
    with gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
