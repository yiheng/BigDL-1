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
package com.intel.analytics.bigdl.utils.tf

import java.nio.ByteOrder

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import Tensorflow._
import BigDLToTensorflow._
import org.tensorflow.framework.{DataType, NodeDef}

/**
 * Wrapper of logic to convert module to tensorflow node definition
 */
trait BigDLToTensorflow {

  /**
   * Convert the module to a tensorflow nodedef
   * @return Mapped nodedef list, the first is the output node
   */
  def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
             byteOrder: ByteOrder, dataFormat: TensorflowDataFormat): Seq[NodeDef]
}

object BigDLToTensorflow {
  private[tf] def processSaveDim(dim: Int, dataFormat: TensorflowDataFormat): Int = {
    if (dataFormat == TensorflowDataFormat.NHWC) {
      if (dim == 2) return 4
      if (dim == 3) return 2
      if (dim == 4) return 3
      dim
    } else {
      dim
    }
  }
}

object ReLUToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder, dataFormat: TensorflowDataFormat): Seq[NodeDef] = {
    require(inputs.length == 1, "Relu only accept one input")

    Seq(relu(inputs(0), module.getName()))
  }
}

object LinearToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder, dataFormat: TensorflowDataFormat): Seq[NodeDef] = {
    require(inputs.length == 1, "Linear only accept one input")
    val linear = module.asInstanceOf[Linear[_]]
    val weight = const(linear.weight.t().contiguous(), linear.getName() + "/weight", byteOrder)
    val weightReader = identity(weight, linear.getName() + "/weightReader")
    val mm = matmul(inputs(0), weightReader, linear.getName() + "matmul")
    val bias = const(linear.bias, linear.getName() + "/bias", byteOrder)
    val biasReader = identity(bias, linear.getName() + "/biasReader")
    val add = biasAdd(mm, biasReader, dataFormat, linear.getName() + "/biasAdd")
    Seq(add, biasReader, bias, mm, weightReader, weight)
  }
}

object SpatialConvolutionToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder, dataFormat: TensorflowDataFormat): Seq[NodeDef] = {
    require(inputs.length == 1, "SpatialConvolution only accept one input")
    val spatialConv = module.asInstanceOf[SpatialConvolution[_]]
    val filter = const(spatialConv.weight, spatialConv.getName() + "/filter", byteOrder)
    val filterReader = identity(filter, spatialConv.getName() + "/filterReader")
    val conv = conv2D(inputs(0), filterReader, spatialConv.strideH, spatialConv.strideW,
      spatialConv.kernelW, spatialConv.kernelH, spatialConv.strideW, spatialConv.strideH,
      dataFormat, spatialConv.getName() + "/conv2D")
    val bias = const(spatialConv.bias, spatialConv.getName() + "/bias", byteOrder)
    val biasReader = identity(bias, spatialConv.getName() + "/biasReader")
    val add = biasAdd(conv, biasReader, dataFormat,
      spatialConv.getName() + "/biasAdd")
    Seq(add, biasReader, bias, conv, filterReader, filter)
  }
}

object SqueezeToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder, dataFormat: TensorflowDataFormat): Seq[NodeDef] = {
    require(inputs.length == 1, "Squeeze only accept one input")
    val sq = module.asInstanceOf[Squeeze[_]]
    Seq(squeeze(inputs(0), sq.dims.map(processSaveDim(_, dataFormat) - 1), sq.getName()))
  }
}

object TanhToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder, dataFormat: TensorflowDataFormat): Seq[NodeDef] = {
    require(inputs.length == 1, "Tanh only accept one input")
    Seq(tanh(inputs(0), module.getName()))
  }
}

object ReshapeToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder, dataFormat: TensorflowDataFormat): Seq[NodeDef] = {
    require(inputs.length == 1, "Reshape only accept one input")
    val rh = module.asInstanceOf[Reshape[_]]
    val size = Tensor[Float](rh.size.length)
    var i = 0
    while(i < rh.size.length) {
      size.setValue(i + 1, rh.size(i))
      i += 1
    }
    val shape = const(size, rh.getName() + "/shape", byteOrder, DataType.DT_INT32)
    val reshapeNode = reshape(inputs(0), shape, rh.getName())
    Seq(reshapeNode, shape)
  }
}

object ViewToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder, dataFormat: TensorflowDataFormat): Seq[NodeDef] = {
    require(inputs.length == 1, "Reshape only accept one input")
    val viewLayer = module.asInstanceOf[View[_]]
    val size = Tensor[Float](viewLayer.sizes.length)
    var i = 0
    while(i < viewLayer.sizes.length) {
      size.setValue(i + 1, viewLayer.sizes(i))
      i += 1
    }
    val shape = const(size, viewLayer.getName() + "/shape", byteOrder, DataType.DT_INT32)
    val reshapeNode = reshape(inputs(0), shape, viewLayer.getName())
    Seq(reshapeNode, shape)
  }
}

object MaxpoolToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder, dataFormat: TensorflowDataFormat): Seq[NodeDef] = {
    require(inputs.length == 1, "Maxpool only accept one input")
    val layer = module.asInstanceOf[SpatialMaxPooling[_]]
    Seq(maxPool(inputs(0), layer.kW, layer.kH, layer.padW, layer.padH,
      layer.dW, layer.dH, dataFormat, layer.getName()))
  }
}

object PaddingToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder, dataFormat: TensorflowDataFormat): Seq[NodeDef] = {
    require(inputs.length == 1, "Padding only accept one input")
    val layer = module.asInstanceOf[Padding[_]]
    val padding = Tensor[Float](1, 2)
    if (layer.pad < 0) {
      padding.setValue(1, 1, -layer.pad)
      padding.setValue(1, 2, 0)
    }
    else {
      padding.setValue(1, 1, 0)
      padding.setValue(1, 2, layer.pad)
    }
    val paddingsNode = const(padding, layer.getName() + "/padding", byteOrder, DataType.DT_INT32)
    val padNode = pad(inputs(0), paddingsNode, layer.getName() + "/output")
    Seq(padNode, paddingsNode)
  }
}

object AvgpoolToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder, dataFormat: TensorflowDataFormat): Seq[NodeDef] = {
    require(inputs.length == 1, "Avgpool only accept one input")
    val layer = module.asInstanceOf[SpatialAveragePooling[_]]
    Seq(avgPool(inputs(0), layer.kW, layer.kH, layer.padW, layer.padH,
      layer.dW, layer.dH, dataFormat, layer.getName()))
  }
}

object SigmoidToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder, dataFormat: TensorflowDataFormat): Seq[NodeDef] = {
    require(inputs.length == 1, "Sigmoid only accept one input")
    Seq(sigmoid(inputs(0), module.getName()))
  }
}

object DropoutToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder, dataFormat: TensorflowDataFormat): Seq[NodeDef] = {
    require(inputs.length == 1, "Dropout only accept one input")
    val layer = module.asInstanceOf[Dropout[_]]
    val shapeNode = shape(inputs(0), layer.getName() + "/shape")
    val rand = randomUniform(shapeNode, layer.getName() + "/random")
    val maxNode = const(Tensor[Float](T(1.0f)), layer.getName() + "/max", byteOrder)
    val minNode = const(Tensor[Float](T(0.0f)), layer.getName() + "/max", byteOrder)
    val sub = subtract(maxNode, minNode, layer.getName() + "/sub")
    val mul = multiply(rand, sub, layer.getName() + "/mul")
    val randOutput = add(minNode, mul, layer.getName() + "/rand_output")
    val keepProb = const(Tensor[Float](T(0.5f)), layer.getName() + "/keep_prob", byteOrder)
    val div1 = realdiv(keepProb, inputs(0), layer.getName() + "/div1")
    val div2 = realdiv(keepProb, randOutput, layer.getName() + "/div2")
    val floorNode = floor(div2, layer.getName() + "/floor")
    val output = multiply(div1, floorNode, layer.getName() + "/output")
    Seq(output, floorNode, div2, div1, keepProb, randOutput, mul, sub, minNode, maxNode,
      rand, shapeNode)
  }
}

object CAddTableToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder, dataFormat: TensorflowDataFormat): Seq[NodeDef] = {
    Seq(addN(inputs, module.getName()))
  }
}

object CMultTableToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder, dataFormat: TensorflowDataFormat): Seq[NodeDef] = {
    require(inputs.length == 2, "Tensorflow only support two tensor multiply together")

    Seq(multiply(inputs(0), inputs(1), module.getName()))
  }
}

object JoinTableToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder, dataFormat: TensorflowDataFormat): Seq[NodeDef] = {
    val layer = module.asInstanceOf[JoinTable[_]]
    Seq(concat(inputs, layer.dimension - 1, layer.getName()))
  }
}

object MeanToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder, dataFormat: TensorflowDataFormat): Seq[NodeDef] = {
    require(inputs.length == 1, "Mean only accept one input")
    val layer = module.asInstanceOf[Mean[_]]
    val dimsTensor = Tensor[Float](layer.dimension)
    dimsTensor.setValue(1, layer.dimension)

    val dims = const(dimsTensor, layer.getName() + "/dims", byteOrder)
    val mean = reduceMean(inputs(0), dims, false, layer.getName() + "/output")
    Seq(mean, dims)
  }
}

object SoftMaxToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder, dataFormat: TensorflowDataFormat): Seq[NodeDef] = {
    require(inputs.length == 1, "Softmax only accept one input")
    Seq(softmax(inputs(0), module.getName()))
  }
}

object LogSoftMaxToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder, dataFormat: TensorflowDataFormat): Seq[NodeDef] = {
    require(inputs.length == 1, "LogSoftmax only accept one input")
    Seq(logSoftmax(inputs(0), module.getName()))
  }
}

object BatchNormToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder, dataFormat: TensorflowDataFormat): Seq[NodeDef] = {
    require(inputs.length == 1, "BatchNorm only accept one input")
    val layer = module.asInstanceOf[SpatialBatchNormalization[_]]
    val stdVar = const(layer.saveStd, layer.getName() + "/std", byteOrder)
    val mean = const(layer.saveMean, layer.getName() + "/mean", byteOrder)
    val scale = const(layer.weight, layer.getName() + "/scale", byteOrder)
    val offset = const(layer.bias, layer.getName() + "/offset", byteOrder)
    val div = realdiv(scale, stdVar, layer.getName() + "/div")
    val mul1 = multiply(inputs(0), div, layer.getName() + "/mul1")
    val mul2 = multiply(scale, div, layer.getName() + "/mul2")
    val sub = multiply(offset, scale, layer.getName() + "/sub")
    val output = add(mul1, sub, layer.getName() + "/output")
    Seq(output, sub, mul2, mul1, div, offset, scale, mean, stdVar)
  }
}