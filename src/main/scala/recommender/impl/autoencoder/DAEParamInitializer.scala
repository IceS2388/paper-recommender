package recommender.impl.autoencoder

import java.util

import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.distribution.Distributions
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer
import org.deeplearning4j.nn.params.DefaultParamInitializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.indexing.NDArrayIndex

/**
  * Author:IceS
  * Date:2019-09-11 14:13:07
  * Description:
  * NONE
  */
class DAEParamInitializer extends DefaultParamInitializer {

  override def numParams(conf: NeuralNetConfiguration): Long = {
    val layerConf = conf.getLayer.asInstanceOf[FeedForwardLayer]
    super.numParams(conf) + DAEParamInitializer.numUsers * layerConf.getNOut
  }

  override def init(conf: NeuralNetConfiguration,
                    paramsView: INDArray,
                    initializeParams: Boolean): util.Map[String, INDArray] = {

    val params = super.init(conf, paramsView, initializeParams)
    val layerConf = conf.getLayer.asInstanceOf[FeedForwardLayer]
    val nIn = layerConf.getNIn
    val nOut = layerConf.getNOut
    val nWeightParams = nIn * nOut
    val nUserWeightParams = DAEParamInitializer.numUsers * nOut
    val userWeightView = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(nWeightParams + nOut, nWeightParams + nOut + nUserWeightParams))
    params.put(DAEParamInitializer.USER_WEIGHT_KEY, this.createUserWeightMatrix(conf, userWeightView, initializeParams))
    conf.addVariable(DAEParamInitializer.USER_WEIGHT_KEY)
    params
  }

  protected def createUserWeightMatrix(
                                        conf: NeuralNetConfiguration,
                                        weightParamView: INDArray,
                                        initializeParameters: Boolean): INDArray = {
    val layerConf = conf.getLayer.asInstanceOf[FeedForwardLayer]
    if (initializeParameters) {
      val dist = Distributions.createDistribution(layerConf.getDist)
      createWeightMatrix(DAEParamInitializer.numUsers, layerConf.getNOut, layerConf.getWeightInit, dist, weightParamView, true)
    }
    else
      createWeightMatrix(DAEParamInitializer.numUsers, layerConf.getNOut, null, null, weightParamView, false)
  }

}

object DAEParamInitializer {
  private val initializer = new DAEParamInitializer()
  val USER_WEIGHT_KEY = "uw"
  val numUsers = 0

  def getInstance(): DAEParamInitializer = initializer

}
