package recommender.impl.autoencoder

import java.util

import org.deeplearning4j.nn.api.{Layer, ParamInitializer}
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport
import org.deeplearning4j.optimize.api.TrainingListener
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray

/**
  * Author:IceS
  * Date:2019-09-11 11:52:06
  * Description:
  * NONE
  */
class DAELayer(builder: DAELayer.Builder = new DAELayer.Builder()) extends FeedForwardLayer {

  override def instantiate(conf: NeuralNetConfiguration,
                           trainingListeners: util.Collection[TrainingListener],
                           layerIndex: Int,
                           layerParamsView: INDArray,
                           initializeParams: Boolean,
                           networkDataType: DataType): Layer = {
    val myCustomLayer = new CDAELayerImp(conf)
    myCustomLayer.setListeners(trainingListeners) //Set the iteration listeners, if any
    myCustomLayer.setIndex(layerIndex) //Integer index of the layer

    //Parameter view array: In Deeplearning4j, the network parameters for the entire network (all layers) are
    // allocated in one big array. The relevant section of this parameter vector is extracted out for each layer,
    // (i.e., it's a "view" array in that it's a subset of a larger array)
    // This is a row vector, with length equal to the number of parameters in the layer
    myCustomLayer.setParamsViewArray(layerParamsView)

    //Initialize the layer parameters. For example,
    // Note that the entries in paramTable (2 entries here: a weight array of shape [nIn,nOut] and biases of shape [1,nOut]
    // are in turn a view of the 'layerParamsView' array.
    val paramTable = initializer().init(conf, layerParamsView, initializeParams)
    myCustomLayer.setParamTable(paramTable)
    myCustomLayer.setConf(conf)
    return myCustomLayer
  }

  override def initializer(): ParamInitializer = ???

  override def getMemoryReport(inputType: InputType): LayerMemoryReport = ???
}

object DAELayer {

  class Builder(val numUsers: Int = 0) extends FeedForwardLayer.Builder {

    override def build[DAELayer](): DAELayer = {
      new DAELayer(this)
    }
  }

}
