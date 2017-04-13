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

import java.io.{DataInputStream, FileInputStream}
import java.nio.ByteBuffer

import org.tensorflow.framework.{GraphDef, NodeDef}
import com.google.protobuf.{ByteString, CodedInputStream}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}

import scala.collection.mutable.Queue
import scala.collection.mutable.ArrayBuffer

/**
  * Created by sujiezhu on 3/31/2017.
  */
class TensorflowLoader {

  private val opNames = ArrayBuffer[String]()

  val layerList  = ArrayBuffer[NodeDef]()
  var nodeNameMap:Map[String,Int] = Map()
  var previousLayer:Map[Int,ArrayBuffer[Int]] = Map()
  var parameter:Map[String,Tensor[Float]] = Map()

  def loadTensorflow(graphPath: String): Unit = {

    // layer operation which we accept
    opNames +=("Conv2D", "Relu","BiasAdd","MaxPool","Reshape","MatMul","Softmax","Argmax","ConcatV2","AvgPool","Squeeze")

    val f = new java.io.File(graphPath)
    val reader = CodedInputStream.newInstance(new DataInputStream(new FileInputStream(f)))
    reader.setSizeLimit(128 << 20)
    require(f.exists(), graphPath + " does not exists")

    val graph = GraphDef.parseFrom(reader)
    val nodes = graph.getNodeList //get the nodelist from the protobuffer file

    var index = 0
    //traverse the nodelist and keep the node we interested in
    for (i <- 0 until nodes.size()){
      if (opNames.contains(nodes.get(i).getOp) || (nodes.get(i).getOp == "Identity" && nodes.get(i).getName.contains("Identity"))){//special case for identity
        val node = nodes.get(i)
        layerList.append(node)
        nodeNameMap += (node.getName->index)
        if (node.getInputCount > 0 && nodeNameMap.contains(node.getInput(0))) {
          val tmp = ArrayBuffer[Int]()
          tmp.append(nodeNameMap(node.getInput(0)))
          previousLayer += (index-> tmp)
          for (j<-1 until node.getInputCount-1){
            if (nodeNameMap.contains(node.getInput(j))){
              previousLayer(index).append(nodeNameMap(node.getInput(j)))
            }
          }
        }
        index += 1
      }
      //deal with the parameters
      if (nodes.get(i).getOp == "Const" && !nodes.get(i).getName.contains("random")){
        System.out.println("Const: " + nodes.get(i).getName)
        val tensor = nodes.get(i).getAttrMap.get("value").getTensor
        val dim = tensor.getTensorShape.getDimCount
        val shape = new Array[Int](dim)
        for(m <-0 until dim){
          shape(m) = tensor.getTensorShape.getDim(m).getSize.toInt
        }

        // deal with the MSB and LSB
        val bf = tensor.getTensorContent
        val buffer = bf.toByteArray
        var tmp: Byte = 0
        for (k <- 0 until buffer.length/4 ){
          tmp = buffer(4 * k)
          buffer(4 * k) = buffer(4 * k + 3)
          buffer(4 * k + 3) = tmp
          tmp = buffer(4 * k + 1)
          buffer(4 * k + 1) = buffer(4 * k + 2)
          buffer(4 * k + 2) = tmp
        }
        val params = ByteBuffer.wrap(buffer).asFloatBuffer
        if (params.capacity > 0) {
          val tmp = new Array[Float](params.capacity())
          for (j <- 0 until params.capacity()){
            tmp(j) = params.get(j)
          }
          val param =Tensor(Storage(tmp)).resize(shape)
          parameter +=(nodes.get(i).getName -> param)
        }
      }
    }

    System.out.println("here to build the graph")
    val g = new Graph
    import scala.collection.JavaConversions._
    for (start <- previousLayer.keySet) {
      for (end <- previousLayer(start)) {
        g.addEdge(end, start)
      }
    }

    System.out.println("Here to traverse the graph")
    val traverse = g.topoSort
    for (order <- traverse) {
      System.out.print(layerList.get(order).getName + "->\n")
      if (previousLayer.contains(order)){
        for(next <- previousLayer(order)){
          System.out.print("Previous layer:  "+layerList.get(next).getName + "->\n")
        }
      }
    }
  }


  class Graph{

    class Vertex(inputvertexlabel: Int) {
      val vertexlabel = inputvertexlabel
      var adjVertex = ArrayBuffer[Vertex]()
      var inDegree = 0
    }

    var directedGraph:Map[Int,Vertex] = Map()

    def addEdge(startNodeLabel: Int, endNodeLabel: Int) {
      var startNode = new Vertex(0)
      var endNode = new Vertex(0)
      if (!directedGraph.contains(startNodeLabel)) {
        startNode = new Vertex(startNodeLabel)
        directedGraph += (startNodeLabel->startNode)
      }
      else{
        startNode = directedGraph(startNodeLabel)
      }
      if (!directedGraph.contains(endNodeLabel)) {
        endNode = new Vertex(endNodeLabel)
        directedGraph += (endNodeLabel->endNode)
      }
      else {
        endNode = directedGraph(endNodeLabel)
      }
      startNode.adjVertex.append(endNode)
      endNode.inDegree += 1
    }

    def topoSort: ArrayBuffer[Integer] = {
      import scala.collection.JavaConversions._
      val result = ArrayBuffer[Integer]()
      val queue = Queue[Vertex]()
      val vertexs = directedGraph.values
      for (vertex <- vertexs) if (vertex.inDegree == 0) queue.enqueue(vertex)
      //if (queue.size>1) throw new Exception
      while (!queue.isEmpty) {
        val tmp = queue.dequeue()
        result.add(tmp.vertexlabel)
        for (v <- tmp.adjVertex) {
          v.inDegree = v.inDegree - 1
          if(v.inDegree == 0){
            queue.enqueue(v)
          }
        }
      }
      result
    }
  }
}


object TensorflowLoader{

  def main(args: Array[String]): Unit = {
    val tensorflowLoader = new TensorflowLoader
    tensorflowLoader.loadTensorflow("./lenet.pb")
  }

  def load(graphPath: String): Unit ={
    val tensorflowLoader = new TensorflowLoader
    tensorflowLoader.loadTensorflow(graphPath)
  }
}