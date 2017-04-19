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

import java.nio.ByteBuffer

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.tensorflow.framework.{DataType, NodeDef}
import TFToBigDL._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}

import scala.collection.mutable.ArrayBuffer

trait TFToBigDL {
  def topology: DirectedGraph[String]

  def layer(tfGraph: DirectedGraph[NodeDef]): (AbstractModule[Activity, Tensor[Float], Float])
}

object FullConnectionTF extends TFToBigDL{
  private val graph = {
    // val add = Node("BiasAdd")
    val add = Node("Add")
    val mul = Node("MatMul")
    Node("*") -> mul
    Node("Const") -> Node("Identity") -> mul -> add
    Node("Const") -> Node("Identity") -> add
    add.graph(reverse = true)
  }
  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef])
      : (AbstractModule[Activity, Tensor[Float], Float]) = {
    val mat = tfGraph.source.prevNodes.filter(node => node.element.getOp.equals("MatMul")).head
    val biasread = tfGraph.source.prevNodes
      .filter(node => node.element.getOp.equals("Identity")).head
    val bias = biasread.prevNodes.filter(node => node.element.getOp.equals("Const")).head
    val weightread = mat.prevNodes.filter(node => node.element.getOp.equals("Identity")).head
    val weights = weightread.prevNodes.filter(node => node.element.getOp.equals("Const")).head

    // get the shape of the bias tensor
    val biasTensor = bias.element.getAttrMap.get("value").getTensor
    val biasdim = biasTensor.getTensorShape.getDimCount
    val biasshape = new Array[Int](biasdim)
    var biassum = 1
    for(m <- 0 until biasdim) {
      biasshape(m) = biasTensor.getTensorShape.getDim(m).getSize.toInt
      biassum = biasshape(m) * biassum
    }
    // get the shape of the weight tensor
    val weightTensor = weights.element.getAttrMap.get("value").getTensor
    val weightdim = weightTensor.getTensorShape.getDimCount
    val weightshape = new Array[Int](weightdim)
    for(m <- 0 until weightdim) {
      weightshape(m) = weightTensor.getTensorShape.getDim(m).getSize.toInt
    }

    val inputSize = weightshape(0)
    val outputSize = weightshape(1)

    val linearLayer = Linear[Float](inputSize, outputSize)
    // transpose the matrix maybe not right
    linearLayer.weight.set(extractParameter(weights.element, weightshape).transpose(1, 2))
    if(biassum != 1) {
      linearLayer.bias.set(extractParameter(bias.element, biasshape))
    } else if (biassum == 1) {
      // for the case that the full connect layer's output dimension is 1
      val biasesArray = new Array[Float](1)
      biasesArray(0) = bias.element.getAttrMap.get("value").getTensor.getFloatVal(0)
      val biases = Tensor(Storage(biasesArray))
      linearLayer.bias.set(biases)
    }

    linearLayer.asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object Conv2D extends TFToBigDL{
  private val graph = {
    val add = Node("BiasAdd")
    val conv = Node("Conv2D")

    Node("*") -> conv -> add
    Node("Const") -> Node("Identity") -> conv -> add
    Node("Const") -> Node("Identity") -> add
    add.graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef])
      : (AbstractModule[Activity, Tensor[Float], Float]) = {
    val conv = tfGraph.source.prevNodes.filter(node => node.element.getOp.equals("Conv2D")).head
    val biasread = tfGraph.source.prevNodes
      .filter(node => node.element.getOp.equals("Identity")).head
    val bias = biasread.prevNodes.filter(node => node.element.getOp.equals("Const")).head
    val weightread = conv.prevNodes.filter(node => node.element.getOp.equals("Identity")).head
    val weights = weightread.prevNodes.filter(node => node.element.getOp.equals("Const")).head

    val strideH = conv.element.getAttrMap.get("strides").getList.getI(2).toInt
    val strideW = conv.element.getAttrMap.get("strides").getList.getI(3).toInt
    var padW = 0
    var padH = 0
    if (conv.element.getAttrMap.get("padding").toString.equals("SAME")) {
      // not support this type right now
      padW = 0
      padH = 0
    }
    else if (conv.element.getAttrMap.get("padding").toString.equals("VALID")) {
      padW = 0
      padH = 0
    }

    // get the shape of the bias tensor
    val biasTensor = bias.element.getAttrMap.get("value").getTensor
    val biasdim = biasTensor.getTensorShape.getDimCount
    val biasshape = new Array[Int](biasdim)
    for (m <- 0 until biasdim) {
      biasshape(m) = biasTensor.getTensorShape.getDim(m).getSize.toInt
    }
    // get the shape of the weight tensor
    val weightTensor = weights.element.getAttrMap.get("value").getTensor
    val weightdim = weightTensor.getTensorShape.getDimCount
    val weightshape = new Array[Int](weightdim)
    for (m <- 0 until weightdim) {
      weightshape(m) = weightTensor.getTensorShape.getDim(m).getSize.toInt
    }

    val nInputPlane = weightshape(2)
    val nOuputPlane = weightshape(3)
    val kernelH = weightshape(0)
    val kernelW = weightshape(1)

    val convLayer = SpatialConvolution[Float](
      nInputPlane, nOuputPlane, kernelW, kernelH, strideW, strideH, padW, padH)
    convLayer.bias.set(extractParameter(bias.element, biasshape))
    convLayer.weight.set(extractParameter(weights.element, weightshape)
      .transpose(1, 4).transpose(2, 3))
    convLayer.asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object ReluTF extends  TFToBigDL {
  private val graph = {
    val nodeinput = new Node("*")
    val nodeRelu = new Node("Relu")
    nodeinput -> nodeRelu
    nodeRelu.graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef])
      : (AbstractModule[Activity, Tensor[Float], Float]) = {
    ReLU[Float]().asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object TanhTF extends  TFToBigDL{
  private val graph = {
    val nodeinput = new Node("*")
    val nodetanh = new Node("Tanh")
    nodeinput -> nodetanh
    nodetanh.graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef])
      : (AbstractModule[Activity, Tensor[Float], Float]) = {
    Tanh[Float]().asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object ReshapeTF extends TFToBigDL {
  private val graph = {
    val nodeinput = new Node("*")
    val nodeReshape = new Node("Reshape")
    val nodeshape = new Node("Const")
    nodeinput -> nodeReshape
    nodeshape -> nodeReshape
    nodeReshape.graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef])
      : (AbstractModule[Activity, Tensor[Float], Float]) = {
    val shape = tfGraph.source.prevNodes.filter(node => node.element.getOp.equals("Const")).head
    val sizeTensor = shape.element.getAttrMap.get("value").getTensor
    val sizedim = sizeTensor.getTensorShape.getDimCount
    val sizeshape = new Array[Int](sizedim)
    for (m <- 0 until sizedim) {
      sizeshape(m) = sizeTensor.getTensorShape.getDim(m).getSize.toInt
    }

    val sizes = extractParameter(shape.element, sizeshape)
    val pos = Array(0, 1)
    val size = Array(sizes(pos).toInt)
    Reshape[Float](size = size)
      .asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object MaxPoolingTF extends TFToBigDL {
  private val graph = {
    val nodeinput = new Node("*")
    val nodeMax = new Node("MaxPool")
    nodeinput -> nodeMax
    nodeMax.graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef])
      : (AbstractModule[Activity, Tensor[Float], Float]) = {
    val maxpool = tfGraph.source.element

    val strideH = maxpool.getAttrMap.get("strides").getList.getI(2).toInt
    val strideW = maxpool.getAttrMap.get("strides").getList.getI(3).toInt

    val ksizeH = maxpool.getAttrMap.get("ksize").getList.getI(2).toInt
    val ksizeW = maxpool.getAttrMap.get("ksize").getList.getI(3).toInt

    var padW = 0
    var padH = 0
    if (maxpool.getAttrMap.get("padding").toString.equals("SAME")) {
      // not support this type right now
      padW = 0
      padH = 0
    }
    else if (maxpool.getAttrMap.get("padding").toString.equals("VALID")) {
      padW = 0
      padH = 0
    }

    SpatialMaxPooling[Float](ksizeW, ksizeH, strideW, strideH, padW, padH)
      .asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object AvgPoolingTF extends TFToBigDL{
  private val graph = {
    val nodeinput = new Node("*")
    val nodeAvg = new Node("AvgPool")
    nodeinput -> nodeAvg
    nodeAvg.graph(reverse = true)
  }
  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef])
      : (AbstractModule[Activity, Tensor[Float], Float]) = {
    val avgpool = tfGraph.source.element

    val strideH = avgpool.getAttrMap.get("strides").getList.getI(2).toInt
    val strideW = avgpool.getAttrMap.get("strides").getList.getI(3).toInt

    val ksizeH = avgpool.getAttrMap.get("ksize").getList.getI(2).toInt
    val ksizeW = avgpool.getAttrMap.get("ksize").getList.getI(3).toInt

    var padW = 0
    var padH = 0
    if (avgpool.getAttrMap.get("padding").toString.equals("SAME")) {
      // not support this type right now
      padW = 0
      padH = 0
    }
    else if (avgpool.getAttrMap.get("padding").toString.equals("VALID")) {
      padW = 0
      padH = 0
    }

    SpatialAveragePooling[Float](ksizeW, ksizeH, strideW, strideH, padW, padH)
      .asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object DropoutTF extends TFToBigDL{
  private val graph = {
    val nodeinput = new Node("*")
    val nodediv = new Node("RealDiv")
    val nodeP = new Node("Const")
    val nodeadd = new Node("Add")
    val noderandom = new Node("Add")
    val nodemin = new Node("Const")
    val nodemax = new Node("Const")
    val nodesub = new Node("Sub")
    val nodeshape = new Node("Const")
    val noderandomuniform = new Node("RandomUniform")
    val nodemul = new Node("Mul")
    val nodefloor = new Node("Floor")
    val nodedrop = new Node("Mul")
    nodeinput -> nodediv -> nodedrop
    nodeP -> nodediv
    nodeP -> nodeadd -> nodefloor -> nodedrop
    nodeshape -> noderandomuniform -> nodemul -> noderandom -> nodeadd
    nodemax -> nodesub -> nodemul
    nodemin -> nodesub
    nodemin -> noderandom
    nodedrop.graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef])
      : (AbstractModule[Activity, Tensor[Float], Float]) = {
    val nodefloor = tfGraph.source.prevNodes.filter(node => node.element.getOp.equals("Floor")).head
    val nodediv = tfGraph.source.prevNodes.filter(node => node.element.getOp.equals("RealDiv")).head
    val nodeadd = nodefloor.prevNodes.filter(node => node.element.getOp.equals("Add")).head
    val nodep = nodediv.prevNodes.filter(node => node.element.getOp.equals("Const")).head


    Dropout[Float](nodep.element.getAttrMap.get("value").getTensor.getFloatVal(0))
      .asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object TFToBigDL {
  def patterns : Array[TFToBigDL] = {
    patternList.toArray
  }

  private[utils] def extractParameter(node : NodeDef, shape : Array[Int],
    bigEndian : Boolean = true)
      : Tensor[Float] = {
    val tensor = node.getAttrMap.get("value").getTensor
    // deal with the MSB and LSB
    val bf = tensor.getTensorContent
    val buffer = bf.toByteArray
    var param = Tensor[Float]
    if (bigEndian) {
      var tmp: Byte = 0
      for (k <- 0 until buffer.length / 4 ) {
        tmp = buffer(4 * k)
        buffer(4 * k) = buffer(4 * k + 3)
        buffer(4 * k + 3) = tmp
        tmp = buffer(4 * k + 1)
        buffer(4 * k + 1) = buffer(4 * k + 2)
        buffer(4 * k + 2) = tmp
      }
    }
    if (tensor.getDtype == DataType.DT_FLOAT) {
      val params = ByteBuffer.wrap(buffer).asFloatBuffer
      if (params.capacity > 0) {
        val tmp = new Array[Float](params.capacity())
        for (j <- 0 until params.capacity()) {
          tmp(j) = params.get(j)
        }
        param = Tensor(Storage(tmp)).resize(shape)
      }
    }
    else if (tensor.getDtype == DataType.DT_INT32) {
      val params = ByteBuffer.wrap(buffer).asIntBuffer
      if (params.capacity > 0) {
        val tmp = new Array[Float](params.capacity())
        for (j <- 0 until params.capacity()) {
          tmp(j) = params.get(j)
        }
        param = Tensor(Storage(tmp)).resize(shape)
      }
    }
    param
  }

  private val patternList : ArrayBuffer[TFToBigDL] = {
    val res = new ArrayBuffer[TFToBigDL]()
    res.append(
      FullConnectionTF, DropoutTF, AvgPoolingTF, MaxPoolingTF, ReshapeTF,
      TanhTF, ReluTF, Conv2D
    )
    res
  }
}
