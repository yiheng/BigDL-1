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

import java.io.{File => JFile}
import java.nio.ByteOrder
import java.util.UUID

import com.intel.analytics.bigdl.dataset.{DistributedDataSet, MiniBatch}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.{DistriOptimizer, Trigger}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import com.intel.analytics.bigdl.numeric.NumericFloat

import scala.sys.process._
import scala.math._

object TensorflowLoaderSpec {
  private val data1 = Array(0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.1f)
  private val data2 = Array(0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.1f)
  private val input1: Tensor[Float] = Tensor[Float](Storage[Float](data1))
  private val input2: Tensor[Float] = Tensor[Float](Storage[Float](data2))
  private val nodeNumber = 4
  private val coreNumber = 4

  Engine.init(nodeNumber, coreNumber, true)

  private val batchSize = 2 * coreNumber

  private val prepareData: Int => (MiniBatch[Float]) = index => {
    val input = Tensor[Float]().resize(batchSize, 10)
    val target = Tensor[Float]().resize(batchSize)
    var i = 0
    while (i < batchSize) {
      if (i % 2 == 0) {
        target.setValue(i + 1, 0.0f)
        input.select(1, i + 1).copy(input1)
      } else {
        target.setValue(i + 1, 0.1f)
        input.select(1, i + 1).copy(input2)
      }
      i += 1
    }
    MiniBatch(input, target)
  }
}

@com.intel.analytics.bigdl.tags.Parallel
class TensorflowLoaderSpec extends TensorflowSpecHelper{

  Logger.getLogger("org").setLevel(Level.WARN)
  Logger.getLogger("akka").setLevel(Level.WARN)

  import TensorflowLoaderSpec._

  var sc: SparkContext = null

  var dataSet: DistributedDataSet[MiniBatch[Float]] = null

  before {
    sc = new SparkContext("local[1]", "RDDOptimizerSpec")

    val rdd = sc.parallelize(1 to (256 * 4), 4).map(prepareData)

    dataSet = new DistributedDataSet[MiniBatch[Float]] {
      override def originRDD(): RDD[_] = rdd

      override def data(train : Boolean): RDD[MiniBatch[Float]] = rdd

      override def size(): Long = 256 * nodeNumber

      override def shuffle(): Unit = {}
    }

    Engine.model.setPoolSize(1)
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

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
    val tfGraph = TensorflowLoader.buildTFGraph(results, Seq("output"))
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
    val model = TensorflowLoader.load(path, Seq("Placeholder"), Seq("output"),
      ByteOrder.LITTLE_ENDIAN, TensorflowDataFormat.NHWC)
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

  "Shared weights" should "be the same instance" in {
    tfCheck()
    (("python " + testScriptsPath("share_weight.py")) !!)
    val resource = getClass().getClassLoader().getResource("tf")
    val path = processPath(resource.getPath()) + JFile.separator +
      "loadTest" + JFile.separator + "share_weight.pb"
    val model = TensorflowLoader.load(path, Seq("Placeholder"), Seq("output"),
      ByteOrder.LITTLE_ENDIAN, TensorflowDataFormat.NHWC)
    val container = model.asInstanceOf[Graph[Float]]
    val l1 = container.modules(1).asInstanceOf[Linear[Float]]
    val l2 = container.modules(3).asInstanceOf[Linear[Float]]
    assert(l1.weight eq l2.weight)
    assert(l1.bias eq l2.bias)
  }

  "Shared weights" should "be the same after running optimizer" in {
    tfCheck()
    (("python " + testScriptsPath("share_weight.py")) !!)
    val resource = getClass().getClassLoader().getResource("tf")
    val path = processPath(resource.getPath()) + JFile.separator +
      "loadTest" + JFile.separator + "share_weight.pb"
    val model = TensorflowLoader.load(path, Seq("Placeholder"), Seq("output"),
      ByteOrder.LITTLE_ENDIAN, TensorflowDataFormat.NCHW)
    val container = model.asInstanceOf[Graph[Float]]

    val optimizer = new DistriOptimizer[Float](container, dataSet, new MSECriterion[Float]())
      .setState(T("learningRate" -> 20.0))
      .setEndWhen(Trigger.maxEpoch(5))
    optimizer.optimize()

    val l1 = container.modules(1).asInstanceOf[Linear[Float]]
    val l2 = container.modules(3).asInstanceOf[Linear[Float]]
    assert(l1.weight == l2.weight)
    assert(l1.bias == l2.bias)
  }

  "static simple rnn " should "have the same inference result as tensorflow" in {
    tfCheck()
    (("python " + testScriptsPath("rnn.py")) !!)
    val resource = getClass().getClassLoader().getResource("tf")
    val path = processPath(resource.getPath()) + JFile.separator +
      "loadTest" + JFile.separator + "rnn.pb"
    val results = TensorflowLoader.parse(path)
    val tfGraph = TensorflowLoader.buildTFGraph(results.subList(0, results.size()-1), Seq("output"))
    val model = TensorflowLoader.buildBigDLModel(tfGraph, Seq("input"),
      Seq("output"),
      ByteOrder.LITTLE_ENDIAN, TensorflowDataFormat.NCHW)
    val input = TensorflowToBigDL.toTensor(results.get(0).getAttrMap.get("value").getTensor,
      ByteOrder.LITTLE_ENDIAN).contiguous()
    val tfResult = TensorflowToBigDL.toTensor(results.get(results.size()-1)
      .getAttrMap.get("value").getTensor, ByteOrder.LITTLE_ENDIAN)
    val bigDLResult = model.forward(input)
    val gradient = Tensor[Float](4, 5).rand()

    tfResult.map( bigDLResult.toTensor, (v1, v2) => {
      assert(abs(v1 - v2) / v1 < 1e-6)
      v2
    })
    model.backward(bigDLResult, gradient)
  }

  "static lstm rnn " should "have the same inference result as tensorflow" in {
    tfCheck()
    (("python " + testScriptsPath("rnn_lstm.py")) !!)
    val resource = getClass().getClassLoader().getResource("tf")
    val path = processPath(resource.getPath()) + JFile.separator +
      "loadTest" + JFile.separator + "lstm.pb"
    val results = TensorflowLoader.parse(path)
    val tfGraph = TensorflowLoader.buildTFGraph(results.subList(0, results.size()-1), Seq("output"))
    val model = TensorflowLoader.buildBigDLModel(tfGraph, Seq("input"),
      Seq("output"),
      ByteOrder.LITTLE_ENDIAN, TensorflowDataFormat.NCHW)
    val input = TensorflowToBigDL.toTensor(results.get(0).getAttrMap.get("value").getTensor,
      ByteOrder.LITTLE_ENDIAN).contiguous()
    val tfResult = TensorflowToBigDL.toTensor(results.get(results.size()-1)
      .getAttrMap.get("value").getTensor, ByteOrder.LITTLE_ENDIAN)
    val bigDLResult = model.forward(input)
    val gradient = Tensor[Float](4, 5).rand()

    tfResult.map( bigDLResult.toTensor, (v1, v2) => {
      assert(abs(v1 - v2)/ v1 < 1e-6, s"$v1, $v2")
      v2
    })
    model.backward(bigDLResult, gradient)
  }

  "Tensorflow lenet" should "be load correctly" in {
    val (tf, bigdl) = testModel("lenet", "LeNet/pool2/MaxPool:0", true)
    val transposed = bigdl.transpose(2, 3).transpose(3, 4)
    tf.almostEqual(transposed, 1e-6) should be(true)
  }

  "Tensorflow Alexnet" should "be load correctly" in {
    val (tf, bigdl) = testModel("alexnet", "alexnet_v2/fc8/squeezed:0", true)
    tf.almostEqual(bigdl, 1e-7) should be(true)
  }

  "TensorFlow vgg_a" should "be load correctly" in {
    val (tf, bigdl) = testModel("vgga", "vgg_a/fc8/squeezed:0", true)
    tf.almostEqual(bigdl, 1e-7) should be(true)
  }

  "TensorFlow vgg_16" should "be load correctly" in {
    val (tf, bigdl) = testModel("vgg16", "vgg_16/fc8/squeezed:0", true)
    tf.almostEqual(bigdl, 1e-7) should be(true)
  }

  "TensorFlow vgg_19" should "be load correctly" in {
    val (tf, bigdl) = testModel("vgg19", "vgg_19/fc8/squeezed:0", true)
    tf.almostEqual(bigdl, 1e-7) should be(true)
  }

  "TensorFlow overfeat" should "be load correctly" in {
    val (tf, bigdl) = testModel("overfeat", "overfeat/fc8/squeezed:0", true)
    tf.almostEqual(bigdl, 1e-7) should be(true)
  }

  "TensorFlow inception_v3" should "be load correctly" in {
    val (tf, bigdl) = testModel("inception_v3", "InceptionV3/Logits/SpatialSqueeze:0", true)
    tf.almostEqual(bigdl, 1e-7) should be(true)
  }

  "TensorFlow resnet_v1" should "be load correctly" in {
    val (tf, bigdl) = testModel("resnet_v1", "resnet_v1_101/SpatialSqueeze:0", true)
    tf.almostEqual(bigdl, 1e-6) should be(true)
  }

  "TensorFlow loader" should "run the python to save the modle and " +
    "have the same inferrence result with tensorflow " +
    "after loading slim inception_resnet_v2" in {
    tfCheck()
    (("python " + testScriptsPath("inception_resnet_v2.py")) !!)
    val resource = getClass().getClassLoader().getResource("tf")
    val path = processPath(resource.getPath()) + JFile.separator +
      "loadTest" + JFile.separator + "inception_resnet_v2_save.pb"
    val results = TensorflowLoader.parse(path)
    val tfGraph = TensorflowLoader.buildTFGraph(results.subList(0, results.size()-2),
      Seq("InceptionResnetV2/Logits/Logits/BiasAdd", "InceptionResnetV2/AuxLogits/Logits/BiasAdd"))
    val model = TensorflowLoader.buildBigDLModel(tfGraph, Seq("input"),
      Seq("InceptionResnetV2/Logits/Logits/BiasAdd", "InceptionResnetV2/AuxLogits/Logits/BiasAdd")
      , ByteOrder.LITTLE_ENDIAN, TensorflowDataFormat.NHWC)
    val input = TensorflowToBigDL.toTensor(results.get(0).getAttrMap.get("value").getTensor,
      ByteOrder.LITTLE_ENDIAN)
      .transpose(2, 4).transpose(3, 4).contiguous()
    val gradient1 = Tensor[Float](2, 1001).rand()
    val gradient2 = Tensor[Float](2, 1001).rand()
    val tfResult1 = TensorflowToBigDL.toTensor(results.get(results.size()-2)
      .getAttrMap.get("value").getTensor, ByteOrder.LITTLE_ENDIAN)
    val tfResult2 = TensorflowToBigDL.toTensor(results.get(results.size()-1)
      .getAttrMap.get("value").getTensor, ByteOrder.LITTLE_ENDIAN)
    val BigDLResult = model.forward(input)

    tfResult1.map( BigDLResult.toTable(1), (v1, v2) => {
      assert(abs(v1 - v2) < 1e-7);
      v2
    })
    tfResult2.map( BigDLResult.toTable(2), (v1, v2) => {
      assert(abs(v1 - v2) < 1e-7);
      v2
    })
    model.backward(input, T(gradient1, gradient2))
  }

  private def testModel(modelName: String, endPoint: String, transInput: Boolean)
  : (Tensor[Float], Tensor[Float]) = {

    tfCheck()
    // Generate command and prepare the temp folder
    val s = JFile.separator
    val modelsFolder = processPath(getClass().getClassLoader().getResource("tf").getPath()) +
      s + "models"
    val modelScript = modelsFolder + s + s"$modelName.py"
    val tmpLocation = java.io.File.createTempFile("tensorflowLoaderTest" + UUID.randomUUID(),
      modelName)
    tmpLocation.delete()
    tmpLocation.mkdir()

    require(runPython(s"$modelScript $tmpLocation $endPoint"), "error when run the model script")

    // Load the model and input/output tensors
    val modelFile = tmpLocation + s + "model.pb"
    val tfNodes = TensorflowLoader.parse(modelFile)
    val tfGraph = TensorflowLoader.buildTFGraph(tfNodes, Seq(endPoint.split(":")(0)))
    val model = TensorflowLoader.buildBigDLModel(tfGraph, Seq("input"), Seq(endPoint.split(":")(0)),
      ByteOrder.LITTLE_ENDIAN, TensorflowDataFormat.NHWC)

    import collection.JavaConverters._
    // Compare the tensor contents
    val tfInputTensor = tfNodes.asScala.filter(_.getName == "input")(0)
      .getAttrMap.get("value").getTensor
    val tfOutputTensor = tfNodes.asScala.filter(_.getName == "output")(0)
      .getAttrMap.get("value").getTensor
    val input = TensorflowToBigDL.toTensor(tfInputTensor,
      ByteOrder.LITTLE_ENDIAN)

    val transposeInput = if (transInput) {
      input.transpose(2, 4).transpose(3, 4).contiguous()
    } else {
      input
    }

    (TensorflowToBigDL.toTensor(tfOutputTensor, ByteOrder.LITTLE_ENDIAN),
      model.forward(transposeInput).toTensor)
  }

  private def processPath(path: String): String = {
    if (path.contains(":")) {
      path.substring(1)
    } else {
      path
    }
  }

  private def testScriptsPath(script: String) : String = {
    val resource = getClass().getClassLoader().getResource("tf")
    processPath(resource.getPath()) + JFile.separator + "loadTest" +
      JFile.separator + script
  }
}
