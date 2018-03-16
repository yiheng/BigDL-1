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
package com.intel.analytics.bigdl.utils.tf.loaders

import java.nio.ByteOrder

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.{CAddTable, Identity => IdentityOps}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.tf.Context
import com.intel.analytics.bigdl.utils.tf.loaders.Utils.getType
import org.tensorflow.framework.{DataType, NodeDef}

import scala.reflect.ClassTag

class Add extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {

    val t = getType(nodeDef.getAttrMap, "T")
    if (t == DataType.DT_FLOAT) {
      new CAddTable[T, Float]()
    } else if (t == DataType.DT_INT32) {
      new CAddTable[T, Int]()
    } else if (t == DataType.DT_INT64) {
      new CAddTable[T, Long]()
    } else {
      throw new UnsupportedOperationException(s"Not support numeric type $t")
    }
  }
}

class RefSwitch extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class IsVariableInitialized extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class RandomUniformInt extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class PlaceholderWithDefault extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class BarrierInsertMany extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class BarrierTakeMany extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class Barrier extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class Gather extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class Where extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class StringJoin extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class AsString extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class StopGradient extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class SparseToDense extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class FixedUnigramCandidateSampler extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class ComputeAccidentalHits extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class FIFOQueueV2 extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class GatherNd extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class Max extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class RandomStandardNormal extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class ScatterNd extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class DestroyTemporaryVariable extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class Assign extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class TemporaryVariable extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class AssignAdd extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class RandomShuffleQueueV2 extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class Cos extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class Sin extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class AdjustContrastv2 extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class HSVToRGB extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class RGBToHSV extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class DepthToSpace extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class BatchToSpaceND extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class SpaceToBatchND extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class NonMaxSuppressionV2 extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class ZerosLike extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class Size extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}

class CropAndResize extends TensorflowOpsLoader {
  override def build[T: ClassTag](nodeDef: NodeDef, byteOrder: ByteOrder
    , context: Context[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    IdentityOps[T]()
  }
}
