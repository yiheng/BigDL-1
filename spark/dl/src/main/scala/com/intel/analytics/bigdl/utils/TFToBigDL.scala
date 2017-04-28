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

import java.nio.charset.Charset
import java.nio.{ByteBuffer, ByteOrder}

import collection.JavaConverters._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.tensorflow.framework.{DataType, NodeDef, TensorProto}
import TFToBigDL._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}

import scala.collection.mutable.ArrayBuffer

trait TFToBigDL {
  def topology: DirectedGraph[String]

  def layer(tfGraph: DirectedGraph[NodeDef]): (AbstractModule[Activity, Tensor[Float], Float])
}

object FullConnectionTF extends TFToBigDL{
  private val graph = {
    val add = Node("BiasAdd")
    val mul = Node("MatMul")
    Node("*") -> mul
    Node("Const") -> Node("Identity") -> mul -> add
    Node("Const") -> Node("Identity") -> add
    add.graph(reverse = true)
  }
  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef])
      : (AbstractModule[Activity, Tensor[Float], Float]) = {
    val bias = TFToBigDL.toTensor(
      tfGraph.source.prevNodes(1).prevNodes(0).element.getAttrMap.get("value").getTensor
    )
    val weight = TFToBigDL.toTensor(
      tfGraph.source.prevNodes(0).prevNodes(1).prevNodes(0).element
        .getAttrMap.get("value").getTensor
    )

    val linearLayer = Linear[Float](weight.size(1), weight.size(2))
    linearLayer.weight.copy(weight.transpose(1, 2))
    linearLayer.bias.copy(bias)
    linearLayer.asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object SqueezeTF extends TFToBigDL {
  private val graph = (Node("*") -> Node("Squeeze")).graph(reverse = true)
  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef])
    : AbstractModule[Activity, Tensor[Float], Float] = {
    val dims = tfGraph.source.element.getAttrOrThrow("squeeze_dims").getList().getIList()
      .asScala.map(_.toInt).toArray
    Squeeze[Float](dims, batchMode = true)
      .asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
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
    val attributes = tfGraph.source.prevNodes(0).element.getAttrMap
    require(attributes.get("strides").getList.getI(0).toInt == 1, s"not support strides on batch")

    val (strideH, strideW) = if (attributes.get("data_format").getS
      .toString(Charset.defaultCharset()) == "NHWC") {
      require(attributes.get("strides").getList.getI(3).toInt == 1, s"not support strides on depth")
      (attributes.get("strides").getList.getI(1).toInt,
        attributes.get("strides").getList.getI(2).toInt)
    } else if (attributes.get("data_format").getS.toString(Charset.defaultCharset()) == "NCHW") {
      require(attributes.get("strides").getList.getI(2).toInt == 1, s"not support strides on depth")
      (attributes.get("strides").getList.getI(2).toInt,
        attributes.get("strides").getList.getI(3).toInt)
    } else {
      throw new IllegalArgumentException("no supported data format")
    }
    val bias = TFToBigDL.toTensor(
      tfGraph.source.prevNodes(1).prevNodes(0).element.getAttrMap.get("value").getTensor)

    val weights = TFToBigDL.toTensor(
      tfGraph.source.prevNodes(0).prevNodes(1).prevNodes(0)
        .element.getAttrMap.get("value").getTensor)
    val nInputPlane = weights.size(3)
    val nOuputPlane = weights.size(4)
    val kernelH = weights.size(1)
    val kernelW = weights.size(2)

    val (pW, pH) =
      if (attributes.get("padding").getS.toString(Charset.defaultCharset()) == "SAME") {
        require((kernelW - strideW) % 2 == 0)
        require((kernelH - strideH) % 2 == 0)
        ((kernelW - strideW) / 2, (kernelH - strideH) / 2)
      } else {
        (0, 0)
      }

    val convLayer = SpatialConvolution[Float](
      nInputPlane, nOuputPlane, kernelW, kernelH, strideW, strideH, pW, pH)
    convLayer.bias.copy(bias)
    convLayer.weight.copy(weights.transpose(1, 4).transpose(2, 3))
    convLayer.asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object ReluTF extends  TFToBigDL {
  private val graph = {
    (Node("*") -> Node("Relu")).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef])
      : (AbstractModule[Activity, Tensor[Float], Float]) = {
    ReLU[Float]().asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object TanhTF extends  TFToBigDL{
  private val graph = {
    (Node("*") -> Node("Tanh")).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef])
      : (AbstractModule[Activity, Tensor[Float], Float]) = {
    Tanh[Float]().asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object ReshapeTF extends TFToBigDL {
  private val graph = {
    val nodeReshape = Node("Reshape")
    Node("*") -> nodeReshape
    Node("Const") -> nodeReshape
    nodeReshape.graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef])
      : (AbstractModule[Activity, Tensor[Float], Float]) = {
    val sizes = TFToBigDL.toTensor(
      tfGraph.source.prevNodes(1).element.getAttrMap.get("value").getTensor)

    val batchMode = sizes.valueAt(1) == -1
    val arraySize = new Array[Int](if (batchMode) sizes.nElement() - 1 else sizes.nElement())
    var i = if (batchMode) 2 else 1
    var k = 0
    while(i <= sizes.nElement()) {
      arraySize(k) = sizes.valueAt(i).toInt
      k += 1
      i += 1
    }
    Reshape[Float](size = arraySize, Some(batchMode))
      .asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object MaxPoolingTF extends TFToBigDL {
  private val graph = {
    (Node("*") -> Node("MaxPool")).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef])
      : (AbstractModule[Activity, Tensor[Float], Float]) = {
    val attributes = tfGraph.source.element.getAttrMap

    val (strideH, strideW, ksizeH, ksizeW) = if (attributes.get("data_format").getS
      .toString(Charset.defaultCharset()) == "NHWC") {
      require(attributes.get("strides").getList.getI(3).toInt == 1, s"not support strides on depth")
      (
        attributes.get("strides").getList.getI(1).toInt,
        attributes.get("strides").getList.getI(2).toInt,
        attributes.get("ksize").getList.getI(1).toInt,
        attributes.get("ksize").getList.getI(2).toInt
      )
    } else if (attributes.get("data_format").getS.toString(Charset.defaultCharset()) == "NCHW") {
      require(attributes.get("strides").getList.getI(2).toInt == 1, s"not support strides on depth")
      (
        attributes.get("strides").getList.getI(2).toInt,
        attributes.get("strides").getList.getI(3).toInt,
        attributes.get("ksize").getList.getI(2).toInt,
        attributes.get("ksize").getList.getI(3).toInt
      )
    } else {
      throw new IllegalArgumentException("no supported data format")
    }

    val (pW, pH) =
      if (attributes.get("padding").getS.toString(Charset.defaultCharset()) == "SAME") {
        require((ksizeW - strideW) % 2 == 0)
        require((ksizeH - strideH) % 2 == 0)
        ((ksizeW - strideW) / 2, (ksizeH - strideH) / 2)
      } else {
        (0, 0)
      }

    val maxpool = SpatialMaxPooling[Float](ksizeW, ksizeH, strideW, strideH, pW, pH)
    maxpool.asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object AvgPoolingTF extends TFToBigDL{
  private val graph = {
    (Node("*") -> Node("AvgPool")).graph(reverse = true)
  }
  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef])
      : (AbstractModule[Activity, Tensor[Float], Float]) = {
    val attributes = tfGraph.source.element.getAttrMap

    val (strideH, strideW, ksizeH, ksizeW) = if (attributes.get("data_format").getS
      .toString(Charset.defaultCharset()) == "NHWC") {
      require(attributes.get("strides").getList.getI(3).toInt == 1, s"not support strides on depth")
      (
        attributes.get("strides").getList.getI(1).toInt,
        attributes.get("strides").getList.getI(2).toInt,
        attributes.get("ksize").getList.getI(1).toInt,
        attributes.get("ksize").getList.getI(2).toInt
      )
    } else if (attributes.get("data_format").getS.toString(Charset.defaultCharset()) == "NCHW") {
      require(attributes.get("strides").getList.getI(2).toInt == 1, s"not support strides on depth")
      (
        attributes.get("strides").getList.getI(2).toInt,
        attributes.get("strides").getList.getI(3).toInt,
        attributes.get("ksize").getList.getI(2).toInt,
        attributes.get("ksize").getList.getI(3).toInt
      )
    } else {
      throw new IllegalArgumentException("no supported data format")
    }

    val (pW, pH) =
      if (attributes.get("padding").getS.toString(Charset.defaultCharset()) == "SAME") {
        require((ksizeW - strideW) % 2 == 0)
        require((ksizeH - strideH) % 2 == 0)
        ((ksizeW - strideW) / 2, (ksizeH - strideH) / 2)
      } else {
        (0, 0)
      }

    SpatialAveragePooling[Float](ksizeW, ksizeH, strideW, strideH, pW, pH)
      .asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object DropoutTF extends TFToBigDL{
  private val graph = {
    val nodediv = Node("RealDiv")
    val nodeP = Node("Const")
    val nodeadd = Node("Add")
    val noderandom = Node("Add")
    val nodemin = Node("Const")
    val nodesub = Node("Sub")
    val nodemul = Node("Mul")
    val nodedrop = Node("Mul")
    Node("*") -> nodediv -> nodedrop
    nodeP -> nodediv
    nodeP -> nodeadd -> Node("Floor") -> nodedrop
    Node("*") -> Node("Shape") -> Node("RandomUniform") -> nodemul -> noderandom -> nodeadd
    Node("Const") -> nodesub -> nodemul
    nodemin -> nodesub
    nodemin -> noderandom
    nodedrop.graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef])
      : (AbstractModule[Activity, Tensor[Float], Float]) = {
    val keepProp = tfGraph.source.prevNodes(0).prevNodes(1).element
      .getAttrMap.get("value").getTensor.getFloatVal(0)

    Dropout[Float](keepProp).asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object Placeholder extends TFToBigDL {
  private val graph = Node("Placeholder").graph(reverse = true)

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef])
      : AbstractModule[Activity, Tensor[Float], Float] = {
    new Input[Float].asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object IdentityTF extends TFToBigDL {
  private val graph = (Node("*") -> Node("Identity")).graph(reverse = true)

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef])
    : AbstractModule[Activity, Tensor[Float], Float] = {
    new Input[Float].asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object BatchNormTF extends TFToBigDL{
  private val graph = {
    val nodeInput = Node("*")
    val nodeMean1 = Node("Mean")
    val nodeStopGrad = Node("StopGradient")
    val nodeSub1 = Node("Sub")
    val nodeSquare = Node("SquaredDifference")
    val nodeMeanss = Node("Sum")
    val nodeVarss = Node("Sum")
    val nodeShape = Node("Reshape")
    val nodeDivisor = Node("Reciprocal")
    val nodeShiftedMean = Node("Mul")
    val nodeMean2 = Node("Add")
    val nodeMul1 = Node("Mul")
    val nodeVariance = Node("Sub")
    val nodeAdd1 = Node("Add")
    val nodeMul2 = Node("Mul")
    val nodeMul3 = Node("Mul")
    val nodeMul4 = Node("Mul")
    val nodeSub2 = Node("Sub")
    val nodeAdd2 = Node("Add")

    nodeInput -> nodeMul3 -> nodeAdd2
    Node("Const") -> Node("Identity") -> nodeSub2
    nodeInput -> nodeMean1 -> nodeStopGrad -> nodeShape
    Node("Const") -> nodeMean1
    nodeInput -> nodeSub1 -> nodeMeanss -> nodeShiftedMean -> nodeMean2 -> nodeMul4
    nodeStopGrad -> nodeSub1
    nodeInput -> nodeSquare -> nodeVarss -> nodeMul1 -> nodeVariance
    nodeStopGrad -> nodeSquare
    Node("Const") -> nodeDivisor -> nodeShiftedMean -> Node("Square") -> nodeVariance -> nodeAdd1
    Node("Const") -> nodeMeanss -> nodeDivisor -> nodeMul1
    Node("Const") -> nodeVarss -> nodeDivisor
    Node("Const") -> nodeAdd1 -> Node("Rsqrt") -> nodeMul2 -> nodeMul3
    Node("Const") -> Node("Identity") -> nodeMul2 -> nodeMul4 -> nodeSub2 -> nodeAdd2
    Node("Const") -> nodeShape -> nodeMean2
    nodeAdd2.graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef])
  : (AbstractModule[Activity, Tensor[Float], Float]) = {
    val nOutput = tfGraph.source.prevNodes(1).prevNodes(1).prevNodes(1)
        .prevNodes(1).prevNodes(0).element.getAttrMap.get("value").getTensor.getIntVal(0)

    val bias = TFToBigDL.toTensor(
      tfGraph.source.prevNodes(1).prevNodes(0).prevNodes(0)
          .element.getAttrMap.get("value").getTensor)
    val weights = TFToBigDL.toTensor(
      tfGraph.source.prevNodes(1).prevNodes(1).prevNodes(1).prevNodes(0).prevNodes(0)
        .element.getAttrMap.get("value").getTensor)

    val spatialBatchNorm = SpatialBatchNormalization[Float](nOutput = nOutput)
    spatialBatchNorm.weight.copy(weights)
    spatialBatchNorm.bias.copy(bias)
    spatialBatchNorm.asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object ConcatTF extends TFToBigDL{
  private val graph = {
    val nodeConcat = new Node("ConcatV2")
    Node("...") -> nodeConcat
    (Node("Const") -> nodeConcat).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef])
  : (AbstractModule[Activity, Tensor[Float], Float]) = {
    val inputNumber = tfGraph.source.element.getAttrMap.get("N").getI.toInt
    val nodeaxis = tfGraph.source.prevNodes(inputNumber)
    val axis = nodeaxis.element.getAttrMap.get("value").getTensor.getIntVal(0)

    val dataFormatMatch = Map("N" -> 0, "H" -> 2, "w" -> 3, "C" -> 1)

    val dimension = dataFormatMatch(TFToBigDL.dataFormat.charAt(axis).toString)
    val nInputDims = 4

    new JoinTable[Float](dimension = dimension + 1, nInputDims = nInputDims)
      .asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object TFToBigDL {

  /**
   * Get the pattern list.
   * @return
   */
  def patterns : Array[TFToBigDL] = {
    patternList.toArray
  }

  /**
   * Switch endianess to big endian. You should do this when you save the model in a big endian
   * environment. The default endianess is little endian.
   */
  def bigEndian : Unit = endian = ByteOrder.BIG_ENDIAN

  /**
   * Switch endianess to little endian. You should do this when you save the model in a little
   * endian environment. This is the default endianess.
   */
  def littleEndian : Unit = endian = ByteOrder.LITTLE_ENDIAN

  /**
   * Register a new mapping from tensor flow operations to BigDL layer. The mapping is defined as
   * a subclass of TFToBigDL, which defines an operation topology(reversed graph) and how to get
   * constructor parameters from the topology.
   * @param pattern
   */
  def registerPattern(pattern : TFToBigDL): Unit = {
    require(pattern.topology.reverse == true, "the topology should be a reversed graph")
    patternList.append(pattern)
    sortPattern()
  }

  private var endian = ByteOrder.LITTLE_ENDIAN

  var dataFormat : String = "NHWC"

  def dataNCHW : Unit = dataFormat = "NCHW"

  /**
   * Convert a tensorflow tensor proto to BigDL tensor
   * @param tfTensor
   * @return
   */
  private[utils] def toTensor(tfTensor: TensorProto): Tensor[Float] = {
    require(tfTensor.getDtype == DataType.DT_FLOAT || tfTensor.getDtype == DataType.DT_INT32,
      s"Data type ${tfTensor.getDtype} is not supported now")
    val shape = tfTensor.getTensorShape.getDimList.asScala.map(_.getSize.toInt).toArray

    if (shape.product == 1) {
      if (tfTensor.getDtype == DataType.DT_FLOAT) {
        return Tensor[Float](T(tfTensor.getFloatVal(0)))
      } else {
        return Tensor[Float](T(tfTensor.getIntVal(0)))
      }
    }

    val buffer = ByteBuffer.wrap(tfTensor.getTensorContent.toByteArray)
    buffer.order(endian)

    if (tfTensor.getDtype == DataType.DT_FLOAT) {
      val params = buffer.asFloatBuffer
      val tmp = new Array[Float](params.capacity())
      var j = 0
      while(j < params.capacity()) {
        tmp(j) = params.get(j)
        j += 1
      }
      Tensor(Storage(tmp), 1, shape)
    } else {
      val params = buffer.asIntBuffer
      val tmp = new Array[Float](params.capacity())
      var j = 0
      while(j < params.capacity()) {
        tmp(j) = params.get(j)
        j += 1
      }
      Tensor(Storage(tmp), 1, shape)
    }
  }

  private var patternList : ArrayBuffer[TFToBigDL] = {
    val res = new ArrayBuffer[TFToBigDL]()
    res.append(
      FullConnectionTF, DropoutTF, AvgPoolingTF, MaxPoolingTF, ReshapeTF,
      TanhTF, ReluTF, Conv2D, Placeholder, SqueezeTF, IdentityTF, ConcatTF, BatchNormTF
    )
    res
  }

  sortPattern()

  /**
   * Sort the pattern list to make sure the graph match first should not be a sub-graph of the graph
   * match later
   */
  private def sortPattern() : Unit = {
    // do not calculate size and edges of a graph every time
    val topToNNodes = patternList.map(g => g -> g.topology.size).toMap
    val topToNEdges = patternList.map(g => g -> g.topology.edges).toMap
    patternList = patternList.sortWith((l, r) => {
      if (topToNNodes(l) != topToNNodes(r)) {
        // graph with more nodes comes first
        topToNNodes(l) > topToNNodes(r)
      } else {
        // same node number, graph with more edges come first
        topToNEdges(l) > topToNEdges(r)
      }
    })
  }
}
