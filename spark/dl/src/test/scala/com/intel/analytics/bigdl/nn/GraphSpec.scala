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
package com.intel.analytics.bigdl.nn

import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{RandomGenerator, T}

@com.intel.analytics.bigdl.tags.Parallel
class GraphSpec extends FlatSpec with Matchers {
  "Graph init" should "throw exceptions when there's cycle" in {
    val fc1 = Linear(4, 2).apply()
    val relu1 = ReLU().apply(fc1)
    relu1 -> fc1

    intercept[IllegalArgumentException] {
      Graph(fc1, relu1)
    }
  }

  "Graph init" should "throw exceptions when some inputs are ignored" in {
    val fc1 = Linear(4, 2).apply()
    val fc2 = Linear(4, 2).apply()
    val output = CAddTable().apply(fc1, fc2)

    intercept[IllegalArgumentException] {
      Graph(fc1, output)
    }
  }

  "Graph init" should "be successful output are ignored" in {
    val fc1 = Linear(4, 2).apply()
    val fc2 = Linear(4, 2).apply()
    val cadd = CAddTable().apply(fc1, fc2)
    val output1 = ReLU().apply(cadd)
    val output2 = ReLU().apply(cadd)

    val graph = Graph(Array(fc1, fc2), Array(output1))
    fc1.element.getParameters()._1.apply1(_ => 1.0f)
    fc2.element.getParameters()._1.apply1(_ => 1.0f)
    val output = graph.forward(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
      Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))))
    output should be(Tensor(T(2.2f, 2.2f)))
  }

  "Graph forward" should "be successful" in {
    val fc1 = Linear(4, 2).apply()
    val fc2 = Linear(4, 2).apply()
    val cadd = CAddTable().apply(fc1, fc2)
    val output1 = ReLU().apply(cadd)
    val output2 = Threshold(10.0).apply(cadd)

    val graph = Graph(Array(fc1, fc2), Array(output1, output2))
    fc1.element.getParameters()._1.apply1(_ => 1.0f)
    fc2.element.getParameters()._1.apply1(_ => 1.0f)
    val output = graph.forward(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
      Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))))
    output should be(T(Tensor(T(2.2f, 2.2f)), Tensor(T(.0f, .0f))))
  }

  "Graph forward" should "be successful when exchange input order" in {
    val fc1 = Linear(4, 2).apply()
    val fc2 = Linear(4, 2).apply()
    val cadd = CAddTable().apply(fc1, fc2)
    val output1 = ReLU().apply(cadd)
    val output2 = Threshold(10.0).apply(cadd)

    val graph = Graph(Array(fc2, fc1), Array(output1, output2))
    fc1.element.getParameters()._1.apply1(_ => 1.0f)
    fc2.element.getParameters()._1.apply1(_ => 2.0f)
    val output = graph.forward(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
      Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))))
    output should be(T(Tensor(T(2.8f, 2.8f)), Tensor(T(0.0f, 0.0f))))
  }

  "Graph forward" should "be successful when exchange output order" in {
    val fc1 = Linear(4, 2).apply()
    val fc2 = Linear(4, 2).apply()
    val cadd = CAddTable().apply(fc1, fc2)
    val output1 = ReLU().apply(cadd)
    val output2 = Threshold(10.0).apply(cadd)

    val graph = Graph(Array(fc1, fc2), Array(output2, output1))
    fc1.element.getParameters()._1.apply1(_ => 1.0f)
    fc2.element.getParameters()._1.apply1(_ => 2.0f)
    val output = graph.forward(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
      Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))))
    output should be(T(Tensor(T(0.0f, 0.0f)), Tensor(T(3.8f, 3.8f))))
  }

  "Graph backward" should "be successful" in {
    val fc1 = Linear(4, 2).apply()
    val fc2 = Linear(4, 2).apply()
    val cadd = CAddTable().apply(fc1, fc2)
    val output1 = ReLU().apply(cadd)
    val output2 = Threshold(10.0).apply(cadd)

    val graph = Graph(Array(fc1, fc2), Array(output1, output2))
    fc1.element.getParameters()._1.apply1(_ => 1.0f)
    fc2.element.getParameters()._1.apply1(_ => 2.0f)
    val output = graph.forward(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
      Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))))
    val gradInput = graph.backward(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
      Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))), T(Tensor(T(1.0f, 2.0f)), Tensor(T(3.0f, 4.0f))))
    gradInput should be(T(Tensor(T(3.0f, 3.0f, 3.0f, 3.0f)),
      Tensor(T(6.0f, 6.0f, 6.0f, 6.0f))))
    fc1.element.parameters()._2(0) should be(Tensor(T(T(0.1f, 0.2f, -0.3f, -0.4f),
      T(0.2f, 0.4f, -0.6f, -0.8f))))
    fc1.element.parameters()._2(1) should be(Tensor(T(1.0f, 2.0f)))
    fc2.element.parameters()._2(0) should be(Tensor(T(T(0.5f, 0.4f, -0.2f, -0.1f),
      T(1.0f, 0.8f, -0.4f, -0.2f))))
    fc2.element.parameters()._2(1) should be(Tensor(T(1.0f, 2.0f)))
  }

  "Graph backward" should "be successful when exchange input order" in {
    val fc1 = Linear(4, 2).apply()
    val fc2 = Linear(4, 2).apply()
    val cadd = CAddTable().apply(fc1, fc2)
    val output1 = ReLU().apply(cadd)
    val output2 = Threshold(10.0).apply(cadd)

    val graph = Graph(Array(fc2, fc1), Array(output1, output2))
    fc1.element.getParameters()._1.apply1(_ => 1.0f)
    fc2.element.getParameters()._1.apply1(_ => 2.0f)
    val output = graph.forward(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
      Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))))
    val gradInput = graph.backward(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
      Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))), T(Tensor(T(1.0f, 2.0f)), Tensor(T(3.0f, 4.0f))))
    gradInput should be(T(Tensor(T(6.0f, 6.0f, 6.0f, 6.0f)), Tensor(T(3.0f, 3.0f, 3.0f, 3.0f))))
    fc1.element.parameters()._2(0) should be(Tensor(T(T(0.5f, 0.4f, -0.2f, -0.1f),
      T(1.0f, 0.8f, -0.4f, -0.2f))))
    fc1.element.parameters()._2(1) should be(Tensor(T(1.0f, 2.0f)))
    fc2.element.parameters()._2(0) should be(Tensor(T(T(0.1f, 0.2f, -0.3f, -0.4f),
      T(0.2f, 0.4f, -0.6f, -0.8f))))
    fc2.element.parameters()._2(1) should be(Tensor(T(1.0f, 2.0f)))
  }

  "Graph backward" should "be successful when exchange output order" in {
    val fc1 = Linear(4, 2).apply()
    val fc2 = Linear(4, 2).apply()
    val cadd = CAddTable().apply(fc1, fc2)
    val output1 = ReLU().apply(cadd)
    val output2 = Threshold(10.0).apply(cadd)

    val graph = Graph(Array(fc1, fc2), Array(output2, output1))
    fc1.element.getParameters()._1.apply1(_ => 1.0f)
    fc2.element.getParameters()._1.apply1(_ => 2.0f)
    val output = graph.forward(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
      Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))))
    val gradInput = graph.backward(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
      Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))), T(Tensor(T(1.0f, 2.0f)), Tensor(T(3.0f, 4.0f))))
    gradInput should be(T(Tensor(T(7.0f, 7.0f, 7.0f, 7.0f)), Tensor(T(14.0f, 14.0f, 14.0f, 14.0f))))
    fc1.element.parameters()._2(0) should be(Tensor(T(T(0.3f, 0.6f, -0.9f, -1.2f),
      T(0.4f, 0.8f, -1.2f, -1.6f))))
    fc1.element.parameters()._2(1) should be(Tensor(T(3.0f, 4.0f)))
    fc2.element.parameters()._2(0) should be(Tensor(T(T(1.5f, 1.2f, -0.6f, -0.3f),
      T(2.0f, 1.6f, -0.8f, -0.4f))))
    fc2.element.parameters()._2(1) should be(Tensor(T(3.0f, 4.0f)))
  }

  "lenet" should "be same with sequential model" in {
    RandomGenerator.RNG.setSeed(1000)
    val seqModel = Sequential().add(Reshape(Array(1, 28, 28)))
      .add(SpatialConvolution(1, 6, 5, 5).setName("conv1_5x5"))
      .add(Tanh())
      .add(SpatialMaxPooling(2, 2, 2, 2))
      .add(Tanh())
      .add(SpatialConvolution(6, 12, 5, 5).setName("conv2_5x5"))
      .add(SpatialMaxPooling(2, 2, 2, 2))
      .add(Reshape(Array(12 * 4 * 4)))
      .add(Linear(12 * 4 * 4, 100).setName("fc1"))
      .add(Tanh())
      .add(Linear(100, 10).setName("fc2"))
      .add(LogSoftMax())

    RandomGenerator.RNG.setSeed(1000)
    val input = Reshape(Array(1, 28, 28)).apply()
    val conv1 = SpatialConvolution(1, 6, 5, 5).apply(input)
    val tanh1 = Tanh().apply(conv1)
    val pool1 = SpatialMaxPooling(2, 2, 2, 2).apply(tanh1)
    val tanh2 = Tanh().apply(pool1)
    val conv2 = SpatialConvolution(6, 12, 5, 5).apply(tanh2)
    val pool2 = SpatialMaxPooling(2, 2, 2, 2).apply(conv2)
    val reshape = Reshape(Array(12 * 4 * 4)).apply(pool2)
    val fc1 = Linear(12 * 4 * 4, 100).apply(reshape)
    val tanh3 = Tanh().apply(fc1)
    val fc2 = Linear(100, 10).apply(tanh3)
    val output = LogSoftMax().apply(fc2)

    val funcModel = Graph(input, output)

    val inputData = Tensor(4, 28 * 28).rand()
    val outputData1 = seqModel.forward(inputData)
    val outputData2 = funcModel.forward(inputData)

    outputData1 should be(outputData2)

    val gradient = Tensor(4, 10).rand()
    val gradientBP1 = seqModel.backward(inputData, gradient)
    val gradientBP2 = funcModel.backward(inputData, gradient)

    gradientBP1 should be(gradientBP2)
    seqModel.getParameters()._2 should be(funcModel.getParameters()._2)
  }
}
