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
import com.intel.analytics.bigdl.utils.T

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
    fc1.value.getParameters()._1.apply1(_ => 1.0f)
    fc2.value.getParameters()._1.apply1(_ => 1.0f)
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
    fc1.value.getParameters()._1.apply1(_ => 1.0f)
    fc2.value.getParameters()._1.apply1(_ => 1.0f)
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
    fc1.value.getParameters()._1.apply1(_ => 1.0f)
    fc2.value.getParameters()._1.apply1(_ => 2.0f)
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
    fc1.value.getParameters()._1.apply1(_ => 1.0f)
    fc2.value.getParameters()._1.apply1(_ => 2.0f)
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
    fc1.value.getParameters()._1.apply1(_ => 1.0f)
    fc2.value.getParameters()._1.apply1(_ => 1.0f)
    val output = graph.forward(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
      Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))))
    val gradInput = graph.backward(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
      Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))), T(Tensor(T(1.0f, 1.0f)), Tensor(T(2.0f, 2.0f))))
    gradInput should be(T(Tensor(T(0.1f, 0.2f, -0.3f, -0.4f)),
      Tensor(T(0.5f, 0.4f, -0.2f, -0.1f))))
    graph.getParameters()._1 should be(Tensor())
    graph.getParameters()._2 should be(Tensor())
  }
}
