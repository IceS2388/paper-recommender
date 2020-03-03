package recommender.impl

import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.graph.MergeVertex
import org.deeplearning4j.nn.conf.layers.misc.ElementWiseMultiplicationLayer
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.{Logger, LoggerFactory}
import recommender._

import scala.util.Random


/**
  * Author:IceS
  * Date:2019-08-12 19:10:11
  * Description:
  * 著名的神经网络协同过滤模型
  * 存储的NCF，候选集为所有物品
  *
  *
  */
case class NCFParams() extends Params {
  override def getName(): String = this.getClass.getSimpleName.replace("Params", "")

  override def toString: String = {
    s"${this.getClass.getSimpleName}\r\n"
  }
}

class NCFRecommender(ap: NCFParams) extends Recommender {

  @transient private lazy val logger: Logger = LoggerFactory.getLogger(this.getClass)

  override def getParams: Params = ap

  //训练集中用户所拥有item
  private var userHasItem: Map[Int, Seq[Rating]] = _


  private var allTrainingItemSet: Set[Int] = _

  ////每个用户所有的物品=训练集中的物品+测试集中的物品
  private var allUserHadItemsMap: Map[Int, Set[Int]] = _

  override def prepare(data: Seq[Rating]): PreparedData = {


    allUserHadItemsMap = data.groupBy(_.user).map(r => {
      (r._1, r._2.map(_.item).toSet)
    })

    //数据分割前
    new PreparedData(data)
  }

  private var newItemVector: collection.Map[Int, linalg.Vector] = _
  private var newUserVector: collection.Map[Int, linalg.Vector] = _

  override def train(data: TrainingData): Unit = {
    require(data.ratings.nonEmpty, "训练数据不能为空！")

    initData(data)


    initNCF(data)


  }


  private def initData(data: TrainingData):Unit = {
    //1.获取训练集中每个用户的观看列表
    userHasItem = data.ratings.groupBy(_.user)

    allTrainingItemSet = data.ratings.map(_.item).distinct.toSet

    //实现新的统计方法
    val userVectors = userHasItem.map(r => {
      val uid = r._1
      var c1 = 0D //1.0
      var c2 = 0D //2.0
      var c3 = 0D //3.0
      var c4 = 0D //4.0
      var c5 = 0D //5.0

      r._2.foreach(r2 => {
        if (r2.rating == 1.0)
          c1 += 1
        else if (r2.rating == 2.0)
          c2 += 1
        else if (r2.rating == 3.0)
          c3 += 1
        else if (r2.rating == 4.0)
          c4 += 1
        else
          c5 += 1
      })

      (uid, Vectors.dense(c1, c2, c3, c4, c5))
    }).toSeq

    //2.生成物品的统计列表
    val itemVectors = data.ratings.groupBy(_.item).map(r => {
      val iid = r._1

      var c1 = 0D //1.0
      var c2 = 0D //2.0
      var c3 = 0D //3.0
      var c4 = 0D //4.0
      var c5 = 0D //5.0

      r._2.foreach(r2 => {
        if (r2.rating == 1.0)
          c1 += 1
        else if (r2.rating == 2.0)
          c2 += 1
        else if (r2.rating == 3.0)
          c3 += 1
        else if (r2.rating == 4.0)
          c4 += 1
        else
          c5 += 1
      })
      (iid, Vectors.dense(c1, c2, c3, c4, c5))
    }).toSeq

    newUserVector = userVectors.toMap
    newItemVector = itemVectors.toMap
  }

  private def initNCF(data: TrainingData):Unit = {
    /** -------------------神经网络--------------------- **/
    val userVS = newUserVector.head._2.size
    val itemVS = newItemVector.head._2.size
    val unionSize = userVS + itemVS


    //1.配置网络结构
    val computationGraphConf = new NeuralNetConfiguration.Builder()
      .seed(123)
      .activation(Activation.RELU)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(new org.nd4j.linalg.learning.config.AdaDelta())
      .l2(1e-3)
      .graphBuilder()
      .addInputs("user_input", "item_input")

      .addLayer("userLayer", new DenseLayer.Builder().nIn(userVS).nOut(userVS).activation(Activation.IDENTITY).build(), "user_input")
      .addLayer("itemLayer", new DenseLayer.Builder().nIn(itemVS).nOut(itemVS).activation(Activation.IDENTITY).build(), "item_input")
      .addLayer("GML", new ElementWiseMultiplicationLayer.Builder().nIn(unionSize).nOut(unionSize).build(), "userLayer", "itemLayer")

      .addVertex("input", new MergeVertex(), "userLayer", "itemLayer")
      .addLayer("MLP4", new DenseLayer.Builder().nIn(unionSize).nOut(4 * unionSize).build(), "input")
      .addLayer("MLP2", new DenseLayer.Builder().nIn(4 * unionSize).nOut(2 * unionSize).build(), "MLP4")
      .addLayer("MLP1", new DenseLayer.Builder().activation(Activation.SIGMOID).nIn(2 * unionSize).nOut(unionSize).build(), "MLP2")

      .addVertex("ncf", new MergeVertex(), "GML", "MLP1")

      .addLayer("out", new OutputLayer.Builder(LossFunction.XENT)
        .nIn(unionSize + unionSize).activation(Activation.SIGMOID)
        .nOut(1).build(), "ncf")
      .setOutputs("out")
      .build()


    ncfModel = new ComputationGraph(computationGraphConf)
    ncfModel.init()


    val size = data.ratings.size

    //2.正样本数据
    logger.info("正样本数据")
    val positiveData = data.ratings.map(r => {
      //构建特征数据
      val userV = newUserVector(r.user)
      val itemV = newItemVector(r.item)

      (Random.nextInt(size),userV,itemV,1F)
    })

    //3.负样本数据
    logger.info("负样本数据")

    val trainingItemSet = data.ratings.map(_.item).distinct.toSet
    val negativeData = userHasItem.flatMap(r => {
      //当前用户拥有的物品
      val userHadSet = r._2.map(_.item).distinct.toSet
      //未分割前的所有物品
      val userHadAllSet = allUserHadItemsMap(r._1)

      //用户测试集中的物品
      val userTestSet = userHadAllSet.diff(userHadSet)
      val negativeSet = trainingItemSet.diff(userHadSet).diff(userTestSet)

      //保证负样本的数量，和正样本数量一致
      val nSet = Random.shuffle(negativeSet).take(userHadSet.size)

      val userV = newUserVector(r._1)
      nSet.map(itemID => {
        val itemV = newItemVector(itemID)
        (Random.nextInt(size),userV,itemV,0F)
      })
    }).toSeq

    //4.数据打散
    logger.info("数据打散")
    val trainTempData = new Array[(Int, linalg.Vector, linalg.Vector, Float)](positiveData.size + negativeData.size)
    var idx = 0
    positiveData.foreach(r => {
      trainTempData(idx) = r
      idx += 1
    })

    negativeData.foreach(r => {
      trainTempData(idx) = r
      idx += 1
    })

    val finalData = trainTempData.sortBy(_._1).map(r => {
      (r._2, r._3, r._4)
    })

    //训练模型
    logger.info("开始训练模型...")
    val userInEmbedding: INDArray = Nd4j.create(finalData.length, userVS)
    val itemInEmbedding: INDArray = Nd4j.create(finalData.length, itemVS)
    val outLabels: INDArray = Nd4j.create(finalData.length, 1)

    finalData.zipWithIndex.foreach({ case ((userIDVector,itemIDVector,label), index) =>
      userInEmbedding.putRow(index,Nd4j.create(userIDVector.toArray))
      itemInEmbedding.putRow(index,Nd4j.create(itemIDVector.toArray))

      outLabels.putScalar(Array[Int](index,0),label)
    })

    ncfModel.setInputs(userInEmbedding,itemInEmbedding)
    ncfModel.setLabels(outLabels)

    ncfModel.fit()

    logger.info("训练模型完成")
  }

  private var ncfModel: ComputationGraph = _


  override def predict(query: Query): PredictedResult = {

    //用户的已经观看列表
    val currentUserSawSet = userHasItem(query.user).map(_.item)
    logger.info(s"已经观看的列表长度为:${currentUserSawSet.size}")


    //筛选相近用户
    logger.info("除了看过的电影，其它电影都作为候选集进行遍历。")
    val candidateItems = allTrainingItemSet.filter(r => {
      currentUserSawSet.nonEmpty && !currentUserSawSet.contains(r)
    })

    logger.info(s"候选列表长度为：${candidateItems.size}")

    val userV = newUserVector(query.user)
    val userInputs=Nd4j.create(candidateItems.size,userV.size)
    val itemSize=newItemVector(1).size
    val itemInputs=Nd4j.create(candidateItems.size,itemSize)

    val indexToItem = candidateItems.zipWithIndex.map({case(itemID,index)=>
      userInputs.putRow(index,Nd4j.create(userV.toArray))
      itemInputs.putRow(index,Nd4j.create(newItemVector(itemID).toArray))
      (index,itemID)
    }).toMap

    val vs: Array[INDArray] = ncfModel.output(userInputs, itemInputs)
    val result=(0 until vs(0).length().toInt ).map(idx=>{
      val score=vs(0).getFloat(idx)
      (indexToItem(idx),score)
    })

    /** ------end------ **/

    val returnResult = result.map(r => {
      ItemScore(r._1, r._2)
    }).toArray.sortBy(_.score).reverse.take(query.num)

    //排序，返回结果
    PredictedResult(returnResult)

  }
}


