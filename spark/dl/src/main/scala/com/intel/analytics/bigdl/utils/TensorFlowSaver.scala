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

import java.io.FileOutputStream

import com.google.protobuf.CodedOutputStream
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.{Dummy, Graph}
import com.intel.analytics.bigdl.optim.DistriOptimizer.getClass
import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Tensor, TensorDataType}
import org.apache.log4j.Logger
import org.tensorflow.framework.TensorShapeProto.Dim
import org.tensorflow.framework._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

object TensorFlowSaver {
  private val logger = Logger.getLogger(getClass)

  /**
   * Save a graph model to protobuf files so that it can be used in tensorflow
   * @param model
   * @param inputs
   * @param path
   * @tparam T
   */
  def saveGraph[T](
      model : Graph[T],
      inputs : Seq[(String, Seq[Int])],
      path: String): Unit = {
    val inputNodeDefs = inputs.map(buildInputPlaceHolder[T](_, model.getNumericType()))
    val inputNodeCache =
      new mutable.HashMap[AbstractModule[Activity, Tensor[T], T], ArrayBuffer[NodeDef]]()
    model.inputs.zip(inputNodeDefs).foreach(n => {
      inputNodeCache(n._1.element) = ArrayBuffer(n._2)
    })

    val graphBuilder = GraphDef.newBuilder()
    inputNodeDefs.foreach(graphBuilder.addNode(_))

    model.executions.foreach(n => {
      val nodeDefs = n.element.toTFDef(inputNodeCache(n.element))
      nodeDefs.foreach(nDef => {
        graphBuilder.addNode(nDef)
      })
      n.nextNodes.foreach(n => {
        val list = inputNodeCache.getOrElse(n.element, ArrayBuffer())
        list.append(nodeDefs(0))
      })
    })

    // Save to file
    val os = new FileOutputStream(path)
    val output = CodedOutputStream.newInstance(os)
    val graph = graphBuilder.build()
    graph.writeTo(output)
    output.flush()
    os.close()
    logger.info(s"Save as tensorflow model file to $path")
  }

  private def buildInputPlaceHolder[T](pair : (String, Seq[Int]),
      dtype : TensorDataType): NodeDef = {
    val nodeDef = NodeDef.newBuilder()
    nodeDef.setOp("Placeholder")
    nodeDef.setName(pair._1)
    addTypeAttr(nodeDef, dtype, "dtype")

    val shape = TensorShapeProto.newBuilder()
    pair._2.foreach(dim => {
      shape.addDim(Dim.newBuilder().setSize(dim))
    })
    nodeDef.putAttr("shape",
      AttrValue.newBuilder().setShape(shape).build()
    )
    nodeDef.build()
  }

  private[bigdl] def addTypeAttr(build: NodeDef.Builder, dtype : TensorDataType,
                                 typeName: String = "T"): Unit = {
    if (dtype == FloatType) {
      build.putAttr(typeName, AttrValue.newBuilder().setType(DataType.DT_FLOAT).build())
    } else if (dtype == DoubleType) {
      build.putAttr(typeName, AttrValue.newBuilder().setType(DataType.DT_DOUBLE).build())
    } else {
      throw new NotImplementedError(s"type $dtype is not supported")
    }
  }
}

