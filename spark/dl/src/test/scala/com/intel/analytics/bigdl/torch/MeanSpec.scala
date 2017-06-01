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

import com.intel.analytics.bigdl.nn.{Mean, MeanMulDim}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator

@com.intel.analytics.bigdl.tags.Serial
class MeanSpec extends TorchSpec {
    def randomn(): Double = RandomGenerator.RNG.normal(-10, 10)

  "An Mean()" should "generate correct output and grad" in {
    torchCheck()
    val layer = new Mean[Double]()
    val input = Tensor[Double](2, 2, 2)
    input.apply1(x => randomn())
    val gradOutput = Tensor[Double](1, 2, 2)
    gradOutput.apply1(x => randomn())

    val start = System.nanoTime()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.Mean()\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    output should be (luaOutput)
    gradInput should be (luaGradInput)

    println("Test case : Mean, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "An Mean(2, 1)" should "generate correct output and grad" in {
    torchCheck()
    val layer = Mean[Double](2, 1)
    val input = Tensor[Double](2, 2, 2)
    input.apply1(x => randomn())
    val gradOutput = Tensor[Double](1, 2, 2)
    gradOutput.apply1(x => randomn())

    val start = System.nanoTime()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.Mean(2,1)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    output should be (luaOutput)
    gradInput should be (luaGradInput)

    println("Test case : Mean, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "A MeanMulDim(Array(2, 3), 3)" should "generate correct output and grad for multi dimension" in {
    val layer = MeanMulDim[Float](Array(2, 3), 3)
    val input = Tensor[Float](1, 3, 3)
    input(Array(1, 1, 1)) = 0.01f
    input(Array(1, 1, 2)) = 0.02f
    input(Array(1, 1, 3)) = 0.03f
    input(Array(1, 2, 1)) = 0.04f
    input(Array(1, 2, 2)) = 0.05f
    input(Array(1, 2, 3)) = 0.06f
    input(Array(1, 3, 1)) = 0.07f
    input(Array(1, 3, 2)) = 0.08f
    input(Array(1, 3, 3)) = 0.09f
    val gradOutput = Tensor[Float](1, 1, 1)
    gradOutput(Array(1, 1, 1)) = 0.09f
    val expectedOutput = Tensor[Float](1, 1, 1)
    expectedOutput(Array(1, 1, 1)) = 0.05f
    val expectedGrad = Tensor[Float](1, 3, 3)
    expectedGrad(Array(1, 1, 1)) = 0.01f
    expectedGrad(Array(1, 1, 2)) = 0.01f
    expectedGrad(Array(1, 1, 3)) = 0.01f
    expectedGrad(Array(1, 2, 1)) = 0.01f
    expectedGrad(Array(1, 2, 2)) = 0.01f
    expectedGrad(Array(1, 2, 3)) = 0.01f
    expectedGrad(Array(1, 3, 1)) = 0.01f
    expectedGrad(Array(1, 3, 2)) = 0.01f
    expectedGrad(Array(1, 3, 3)) = 0.01f
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
