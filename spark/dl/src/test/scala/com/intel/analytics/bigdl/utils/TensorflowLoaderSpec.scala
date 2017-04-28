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

import java.io.{File => JFile}

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}
import org.tensorflow.framework.NodeDef

@com.intel.analytics.bigdl.tags.Parallel
class TensorflowLoaderSpec extends FlatSpec with Matchers {
  "TensorFlow loader" should "read a list of nodes from pb file" in {
    val resource = getClass().getClassLoader().getResource("tf")
    val path = processPath(resource.getPath()) + JFile.separator + "test.pb"
    val results = TensorflowLoader.parse(path)
    results.size() should be(14)
  }

  "TensorFlow loader" should "be able to build a TF graph" in {
    val resource = getClass().getClassLoader().getResource("tf")
    val path = processPath(resource.getPath()) + JFile.separator + "test.pb"
    val results = TensorflowLoader.parse(path)
    val tfGraph = TensorflowLoader.buildTFGraph(results)
    tfGraph.size should be(15)  // there's a dummy output
    val topSort = tfGraph.topologySort// It can do topology sort
    topSort.length should be(15)
    topSort(0).element should be(null)
    topSort(1).element.getName should be("output")
    topSort(2).element.getName should be("MatMul_1")
    topSort(3).element.getName should be("Variable_3/read")
    topSort(4).element.getName should be("Variable_3")
    topSort(5).element.getName should be("Tanh")
    topSort(6).element.getName should be("Variable_2/read")
    topSort(7).element.getName should be("Variable_2")
    topSort(8).element.getName should be("BiasAdd")
    topSort(9).element.getName should be("MatMul")
    topSort(10).element.getName should be("Variable_1/read")
    topSort(11).element.getName should be("Variable_1")
    topSort(12).element.getName should be("Placeholder")
    topSort(13).element.getName should be("Variable/read")
    topSort(14).element.getName should be("Variable")
  }

  "TensorFlow loader" should "be able to build a BigDL graph" in {
    val resource = getClass().getClassLoader().getResource("tf")
    val path = processPath(resource.getPath()) + JFile.separator + "test.pb"
    val results = TensorflowLoader.parse(path)
    val tfGraph = TensorflowLoader.buildTFGraph(results)
    val model = TensorflowLoader.buildBigDLModel(tfGraph, Seq("Placeholder"), Seq("output"))
    val container = model.asInstanceOf[Graph[Float]]
    container.modules.length should be(4)
    RandomGenerator.RNG.setSeed(100)
    val input = Tensor[Float](4, 1).rand()
    val output1 = container.forward(input)

    val model2 = Sequential[Float]()
    val fc1 = Linear[Float](1, 10)
    fc1.parameters()._1(0).fill(0.2f)
    fc1.parameters()._1(1).fill(0.1f)
    model2.add(fc1).add(Tanh())

    val fc2 = Linear[Float](10, 1)
    fc2.parameters()._1(0).fill(0.2f)
    fc2.parameters()._1(1).fill(0.1f)
    model2.add(fc2)

    val output2 = model2.forward(input)
    output1 should be(output2)
  }

  "TensorFlow loader" should "be able to load slim alexnetv2" in {
    val resource = getClass().getClassLoader().getResource("tf")
    val path = processPath(resource.getPath()) + JFile.separator + "alexnet.pb"
    val results = TensorflowLoader.parse(path)
    val tfGraph = TensorflowLoader.buildTFGraph(results)
    val model = TensorflowLoader.buildBigDLModel(tfGraph, Seq("Placeholder"),
      Seq("alexnet_v2/fc8/squeezed"))
    val input = Tensor[Float](4, 3, 224, 224).rand()
    val gradient = Tensor[Float](4, 1000).rand()
    model.forward(input)
    model.backward(input, gradient)
  }

  "TensorFlow loader" should "be able to load slim vgga" in {
    val resource = getClass().getClassLoader().getResource("tf")
    val path = processPath(resource.getPath()) + JFile.separator + "vgga.pb"
    val results = TensorflowLoader.parse(path)
    val tfGraph = TensorflowLoader.buildTFGraph(results)
    val model = TensorflowLoader.buildBigDLModel(tfGraph, Seq("Placeholder"),
      Seq("vgg_a/fc8/squeezed"))
    val input = Tensor[Float](4, 3, 224, 224).rand()
    val gradient = Tensor[Float](4, 1000).rand()
    model.forward(input)
    model.backward(input, gradient)
  }

  "TensorFlow loader" should "be able to load slim vgg16" in {
    val resource = getClass().getClassLoader().getResource("tf")
    val path = processPath(resource.getPath()) + JFile.separator + "vgg16.pb"
    val results = TensorflowLoader.parse(path)
    val tfGraph = TensorflowLoader.buildTFGraph(results)
    val model = TensorflowLoader.buildBigDLModel(tfGraph, Seq("Placeholder"),
      Seq("vgg_16/fc8/squeezed"))
    val input = Tensor[Float](4, 3, 224, 224).rand()
    val gradient = Tensor[Float](4, 1000).rand()
    model.forward(input)
    model.backward(input, gradient)
  }

  "TensorFlow loader" should "be able to load slim vgg19" in {
    val resource = getClass().getClassLoader().getResource("tf")
    val path = processPath(resource.getPath()) + JFile.separator + "vgg19.pb"
    val results = TensorflowLoader.parse(path)
    val tfGraph = TensorflowLoader.buildTFGraph(results)
    val model = TensorflowLoader.buildBigDLModel(tfGraph, Seq("Placeholder"),
      Seq("vgg_19/fc8/squeezed"))
    val input = Tensor[Float](2, 3, 224, 224).rand()
    val gradient = Tensor[Float](2, 1000).rand()
    model.forward(input)
    model.backward(input, gradient)
  }

  "TensorFlow loader" should "be able to load slim lenet" in {
    val resource = getClass().getClassLoader().getResource("tf")
    val path = processPath(resource.getPath()) + JFile.separator + "lenet.pb"
    val results = TensorflowLoader.parse(path)
    val tfGraph = TensorflowLoader.buildTFGraph(results)
    val model = TensorflowLoader.buildBigDLModel(tfGraph, Seq("Placeholder"),
      Seq("LeNet/fc4/BiasAdd"))
    val input = Tensor[Float](4, 3, 32, 32).rand()
    val gradient = Tensor[Float](4, 10).rand()
    model.forward(input)
    model.backward(input, gradient)
  }

  "TensorFlow loader" should "be able to load slim inception_v3" in {
    val resource = getClass().getClassLoader().getResource("tf")
    val path = processPath(resource.getPath()) + JFile.separator + "inception_v3.pb"
    val results = TensorflowLoader.parse(path)
    val tfGraph = TensorflowLoader.buildTFGraph(results)
    val model = TensorflowLoader.buildBigDLModel(tfGraph, Seq("Placeholder"),
      Seq("InceptionV3/Logits/SpatialSqueeze"))
    val input = Tensor[Float](2, 3, 299, 299).rand()
    val gradient = Tensor[Float](2, 1000).rand()
    model.forward(input)
    model.backward(input, gradient)
  }

  private def processPath(path: String): String = {
    if (path.contains(":")) {
      path.substring(1)
    } else {
      path
    }
  }
}
