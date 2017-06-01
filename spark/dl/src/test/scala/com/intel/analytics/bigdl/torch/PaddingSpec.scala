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
package com.intel.analytics.bigdl.torch

import com.intel.analytics.bigdl.nn.Padding
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class PaddingSpec extends TorchSpec {


  "A Padding Module " should "generate correct output and grad with nInputDim != input.dim()" in {
    torchCheck()
    val dim = 1
    val pad = -1
    val nInputDim = 4
    val value = -0.8999761
    val index = 14

    val input = Tensor[Double](3, 13, 11).apply1(e => Random.nextDouble())
    val gradOutput = Tensor[Double](3, 14, 11).apply1(e => Random.nextDouble())

    val code = "module = nn.Padding(" + dim + "," + pad + "," + nInputDim + "," +
      value + "," + index + ")\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)\n"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))

    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    val module = Padding[Double](dim, pad, nInputDim, value, index)
    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    output should be(luaOutput1)
    gradInput should be(luaOutput2)

    println("Test case : Padding, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A Padding Module " should "generate correct output and grad with nInputDim == input.dim()" in {
    torchCheck()
    val dim = 1
    val pad = 1
    val nInputDim = 3
    val value = 1
    val index = 2

    val input = Tensor[Double](3, 13, 11).apply1(e => Random.nextDouble())
    val gradOutput = Tensor[Double](5, 13, 11).apply1(e => Random.nextDouble())

    val code = "module = nn.Padding(" + dim + "," + pad + "," + nInputDim + "," +
      value + "," + index + ")\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)\n"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))

    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    val module = Padding[Double](dim, pad, nInputDim, value, index)
    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    output should be(luaOutput1)
    gradInput should be(luaOutput2)

    println("Test case : Padding, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A Padding Module " should "generate correct output and grad with index == 1" in {
    torchCheck()
    val dim = 1
    val pad = -1
    val nInputDim = 4
    val value = -0.8999761
    val index = 1

    val input = Tensor[Double](3, 13, 11).apply1(e => Random.nextDouble())
    val gradOutput = Tensor[Double](3, 14, 11).apply1(e => Random.nextDouble())

    val code = "module = nn.Padding(" + dim + "," + pad + "," + nInputDim + "," +
      value + "," + index + ")\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)\n"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))

    val luaOutput1 = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaOutput2 = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    val module = Padding[Double](dim, pad, nInputDim, value, index)
    val start = System.nanoTime()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    output should be(luaOutput1)
    gradInput should be(luaOutput2)

    println("Test case : Padding, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A Padding(Array(Array(2, 3), Array(-1, 1, -2, 2))" should
    "generate correct output and grad for multi dimension" in {
    val layer = Padding[Float](Array(2, 3), Array(-1, 1, -1, 1), 3, 0.0, 1)
    val input = Tensor[Float](1, 2, 2)
    input(Array(1, 1, 1)) = 0.01f
    input(Array(1, 1, 2)) = 0.02f
    input(Array(1, 2, 1)) = 0.03f
    input(Array(1, 2, 2)) = 0.04f
    val expectedOutput = Tensor[Float](1, 4, 4)
    expectedOutput(Array(1, 1, 1)) = 0.00f
    expectedOutput(Array(1, 1, 2)) = 0.00f
    expectedOutput(Array(1, 1, 3)) = 0.00f
    expectedOutput(Array(1, 1, 4)) = 0.00f
    expectedOutput(Array(1, 2, 1)) = 0.00f
    expectedOutput(Array(1, 2, 2)) = 0.01f
    expectedOutput(Array(1, 2, 3)) = 0.02f
    expectedOutput(Array(1, 2, 4)) = 0.00f
    expectedOutput(Array(1, 3, 1)) = 0.00f
    expectedOutput(Array(1, 3, 2)) = 0.03f
    expectedOutput(Array(1, 3, 3)) = 0.04f
    expectedOutput(Array(1, 3, 4)) = 0.00f
    expectedOutput(Array(1, 4, 1)) = 0.00f
    expectedOutput(Array(1, 4, 2)) = 0.00f
    expectedOutput(Array(1, 4, 3)) = 0.00f
    expectedOutput(Array(1, 4, 4)) = 0.00f

    val gradOutput = Tensor[Float](1, 4, 4)
    gradOutput(Array(1, 1, 1)) = 0.01f
    gradOutput(Array(1, 1, 2)) = 0.02f
    gradOutput(Array(1, 1, 3)) = 0.03f
    gradOutput(Array(1, 1, 4)) = 0.04f
    gradOutput(Array(1, 2, 1)) = 0.05f
    gradOutput(Array(1, 2, 2)) = 0.06f
    gradOutput(Array(1, 2, 3)) = 0.07f
    gradOutput(Array(1, 2, 4)) = 0.08f
    gradOutput(Array(1, 3, 1)) = 0.09f
    gradOutput(Array(1, 3, 2)) = 0.10f
    gradOutput(Array(1, 3, 3)) = 0.11f
    gradOutput(Array(1, 3, 4)) = 0.12f
    gradOutput(Array(1, 4, 1)) = 0.13f
    gradOutput(Array(1, 4, 2)) = 0.14f
    gradOutput(Array(1, 4, 3)) = 0.15f
    gradOutput(Array(1, 4, 4)) = 0.16f
    val expectedGrad = Tensor[Float](1, 2, 2)
    expectedGrad(Array(1, 1, 1)) = 0.06f
    expectedGrad(Array(1, 1, 2)) = 0.07f
    expectedGrad(Array(1, 2, 1)) = 0.10f
    expectedGrad(Array(1, 2, 2)) = 0.11f

    val inputOrg = input.clone()
    val gradOutputOrg = gradOutput.clone()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)

    expectedOutput should be (output)
    expectedGrad should be (gradInput)
    input should be (inputOrg)
    gradOutput should be (gradOutputOrg)
  }
}

