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

package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class DummyCAddTable[T: ClassTag](val index: Int)(
  implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[T], T] {

  // no related with format
  override def updateOutput(input: Table): Tensor[T] = {
    this.output = input[Tensor[T]](index)
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]) : Table = {
    throw new UnsupportedOperationException(s"DummyCAddTable only supports inference/prediction")
  }
}


object DummyCAddTable {
  def apply[@specialized(Float, Double) T: ClassTag](index: Int)(
    implicit ev: TensorNumeric[T]) : DummyCAddTable[T] = {
    new DummyCAddTable[T](index)
  }
}

