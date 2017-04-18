/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.utils

import java.io.{DataInputStream, FileInputStream}

import org.tensorflow.framework.{GraphDef, NodeDef}
import com.google.protobuf.CodedInputStream
import java.util.List

object TensorflowLoader{

  /**
   * Parse a tensor flow model protobuf file, read a list of op nodes from it
   * @param graphPrototxt where is the tf protobuf file
   * @return
   */
  private[bigdl] def load(graphPrototxt: String) : List[NodeDef] = {
    val f = new java.io.File(graphPrototxt)
    val reader = CodedInputStream.newInstance(new DataInputStream(new FileInputStream(f)))
    reader.setSizeLimit(128 << 20)
    require(f.exists(), graphPrototxt + " does not exists")

    val graph = GraphDef.parseFrom(reader)
    graph.getNodeList
  }
}