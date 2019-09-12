package recommender.impl.autoencoder

import java.util

import org.deeplearning4j.nn.api.{Layer, ParamInitializer}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.BaseLayer
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport
import org.deeplearning4j.nn.conf.{InputPreProcessor, NeuralNetConfiguration}
import org.deeplearning4j.optimize.api.TrainingListener
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray

/**
  * Author:IceS
  * Date:2019-09-11 14:13:23
  * Description:
  * NONE
  */
class DAELayerImp(conf:NeuralNetConfiguration) extends BaseLayer[DAELayer]{
  override def instantiate(conf: NeuralNetConfiguration,
                           trainingListeners: util.Collection[TrainingListener],
                           layerIndex: Int,
                           layerParamsView: INDArray,
                           initializeParams: Boolean,
                           networkDataType: DataType): Layer ={
      val layer= super.instantiate(conf,trainingListeners,layerIndex,layerParamsView,initializeParams,networkDataType)
    //TODO
    //layer.

    layer
  }

  override def initializer(): ParamInitializer = ???

  override def getOutputType(layerIndex: Int, inputType: InputType): InputType = ???

  override def setNIn(inputType: InputType, `override`: Boolean): Unit = ???

  override def getPreProcessorForInputType(inputType: InputType): InputPreProcessor = ???

  override def isPretrainParam(paramName: String): Boolean = ???

  override def getMemoryReport(inputType: InputType): LayerMemoryReport = ???
}
object DAELayerImp extends BaseLayer.Builder[DAELayer]{
  override def build[DAELayer](): DAELayer = {

  }
}