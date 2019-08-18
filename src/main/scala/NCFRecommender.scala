import org.apache.spark.mllib.clustering.BisectingKMeans
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.graph.MergeVertex
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.{Logger, LoggerFactory}




/**
  * Author:IceS
  * Date:2019-08-12 19:10:11
  * Description:
  * 著名的神经网络协同过滤模型
  * 1.编码转换，根据评分向量，直接作为稀疏向量。
  * 用户ID  索引   评分
  * userid:itemid,rating
  * 2.聚类.
  * 3.根据中心点，生成新的用户(物品)特征向量
  * 4.根据新的特征向量，搭建神经网络。
  *
  *
  */
// TODO 用随机森林测试
case class NCFParams(userThreashold: Int = 20,
                     itemThreashold: Int = 2,
                     method: String = "Cosine",
                     k: Int = 5,
                     maxIterations: Int = 20) extends Params {
  override def getName(): String = this.getClass.getSimpleName.replace("Params", "")

  override def toString: String = {
    s"${this.getClass.getSimpleName}:{userThreashold:$userThreashold,itemThreashold:$itemThreashold,method:$method,k:$k,maxIterations:$maxIterations\r\n"
  }
}

class NCFRecommender(ap: NCFParams) extends Recommender {

  @transient private lazy val logger: Logger = LoggerFactory.getLogger(this.getClass)
  override def getParams: Params = ap

  private var userId2Index: BiMap[Int, Long] = _
  private var itemId2Index: BiMap[Int, Long] = _

  override def prepare(data: Seq[Rating]): PrepairedData = {
    //生成全部用户ID
    val userIDSet: Seq[Int] = data.map(_.user).distinct
    userId2Index = BiMap.toIndex(userIDSet)

    //生成全部物品ID
    val itemIDSet = data.map(_.item).distinct
    itemId2Index = BiMap.toIndex(itemIDSet)

    new PrepairedData(data)
  }



  override def train(trainingData: TrainingData): Unit = {
    //所有物品的大小
    val userGroup = trainingData.ratings.groupBy(_.user)
    userHasItem = userGroup

    //1.对用户聚类。直接用稀疏矩阵尝试
    val userVector: Map[Int, linalg.Vector] = userGroup.map(r => {
      //r._1//userId
      val itemSeq = r._2.map(r2 => {
        (itemId2Index(r2.item).toInt, r2.rating)
      })
      (r._1, Vectors.sparse(itemId2Index.size, itemSeq))
    })

    val itemGroup = trainingData.ratings.groupBy(_.item)
    val itemVector: Map[Int, linalg.Vector] = itemGroup.map(r => {
      //r._1//itemId
      val userSeq = r._2.map(r2 => {
        (userId2Index(r2.user).toInt, r2.rating)
      })
      (r._1, Vectors.sparse(userId2Index.size, userSeq))
    })

    //2.根据聚类的中心，生成新向量。
    val sparkSession = SparkSession.builder().master("local[*]").appName(this.getClass.getSimpleName).getOrCreate()

    val bkmUser = new BisectingKMeans().setK(ap.k).setMaxIterations(ap.maxIterations)
    val rddUserVector = sparkSession.sparkContext.parallelize(userVector.values.toSeq)
    val userModel = bkmUser.run(rddUserVector)
    //调试信息
    userModel.clusterCenters.foreach(println)
    //生成新特征向量

    newUserVector = userVector.map(r => {
      //r._1//userID
      val distance: Array[Double] = userModel.clusterCenters.map(c => {
        Correlation.getDistance(r._2, c)
      })
      val sum = distance.sum
      val fdistance = distance.map(r2 => {
        r2 / sum
      })
      (r._1, fdistance)
    })

    val bkmItem = new BisectingKMeans().setK(ap.k).setMaxIterations(ap.maxIterations)
    val rddItemVector = sparkSession.sparkContext.parallelize(itemVector.values.toSeq)
    val itemModel = bkmItem.run(rddItemVector)
    //调试信息
    itemModel.clusterCenters.foreach(println)

    newItemVector = itemVector.map(r => {
      //r._1//userID
      val distance: Array[Double] = itemModel.clusterCenters.map(c => {
        Correlation.getDistance(r._2, c)
      })
      val sum = distance.sum
      val fdistance = distance.map(r2 => {
        r2 / sum
      })
      (r._1, fdistance)
    })

    sparkSession.close()

    //3.搭建神经网络
    //配置网络结构
    val computationGraphConf = new NeuralNetConfiguration.Builder()
      .seed(123)
      .activation(Activation.SIGMOID)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(new org.nd4j.linalg.learning.config.AdaDelta())
      .l2(1e-3)
      .graphBuilder()
      .addInputs("input")
      .addLayer("GML", new DenseLayer.Builder().nIn(2*ap.k).nOut(1).build(), "input")
      .addLayer("MLP4", new DenseLayer.Builder().nIn(2*ap.k).nOut(4 * (2*ap.k)).build(), "input")
      .addLayer("MLP2", new DenseLayer.Builder().nIn(4 * (2*ap.k)).nOut(2 * (2*ap.k)).build(), "MLP4")
      .addLayer("MLP1", new DenseLayer.Builder().nIn(2 * (2*ap.k)).nOut(2*ap.k).build(), "MLP2")
      .addVertex("ncf", new MergeVertex(), "GML", "MLP1")
      .addLayer("out",new OutputLayer.Builder().nIn(2*ap.k+1).nOut(10).build(),"ncf")
      .setOutputs("out")
      .build()

    import org.deeplearning4j.nn.graph.ComputationGraph
    model = new ComputationGraph(computationGraphConf)
    model.init()

    //准备训练数据
    val td = trainingData.ratings.map(r => {
      //构建数据
      val userV = newUserVector(r.user)
      val itemV = newItemVector(r.item)
      val arr = new Array[Float](userV.length + itemV.length)

      userV.indices.foreach(idx => {
        arr(idx) = userV(idx).toFloat
      })

      itemV.indices.foreach(idx => {
        arr(idx + userV.length) = itemV(idx).toFloat
      })

      //生成标签
      val la=new Array[Float](10)
      if(r.rating==0.5)
        la(0)=1
      else if(r.rating==1.0)
        la(1)=1
      else if(r.rating==1.5)
        la(2)=1
      else if(r.rating==2.0)
        la(3)=1
      else if(r.rating==2.5)
        la(4)=1
      else if(r.rating==3.0)
        la(5)=1
      else if(r.rating==3.5)
        la(6)=1
      else if(r.rating==4.0)
        la(7)=1
      else if(r.rating==4.5)
        la(8)=1
      else if(r.rating==5.0)
        la(9)=1
      else
        throw new Exception("评分错误!")

      (Nd4j.create(arr).reshape(1,10), Nd4j.create(la).reshape(1,10))
    })

    td.foreach(r=>{

      model.fit(Array(r._1),Array( r._2))
    })
    //model.fit(td.map(_._1).toArray,td.map(_._2).toArray)

  }

  private var model: ComputationGraph = _
  //训练集中用户所拥有item
  private var userHasItem: Map[Int, Seq[Rating]] = _

  private var newItemVector: Map[Int, Array[Double]] = _
  private var newUserVector: Map[Int, Array[Double]] = _

  override def predict(query: Query): PredictedResult = {

    //判断有没有用户向量
    if(!newUserVector.contains(query.user)){
      //该用户没有最相似的用户列表
      logger.warn(s"该用户:${query.user}没有相似用户列表，无法生成推荐！")
      return PredictedResult(Array.empty)
    }

    //生成候选物品列表
    val userHad=userHasItem(query.user).map(_.item).toSet
    val items= itemId2Index.toMap.keys.filter(r=>{
      userHad.nonEmpty && !userHad.contains(r)
    })

    //根据当前用户的ID和物品列表生成预测集
    val userV =newUserVector(query.user)//用户特征向量
    val result = items.filter(r=>{
      newItemVector.contains(r)
    }).map(r=>{

      val itemV =newItemVector(r)
      val arr = new Array[Float](userV.length + itemV.length)
      userV.indices.foreach(idx => {
        arr(idx) = userV(idx).toFloat
      })

      itemV.indices.foreach(idx => {
        arr(idx + userV.length) = itemV(idx).toFloat
      })

      val vs: Array[INDArray] =model.output(Nd4j.create(arr).reshape(1,10))
      //logger.info(s"vs.length:${vs.length},vs(0).length:${vs(0).length()}")

      val scores=vs(0).getFloat(9)*5.0+vs(0).getFloat(8)*4.5+vs(0).getFloat(7)*4.0+vs(0).getFloat(6)*3.5

      ItemScore(r, scores)
    }).toArray.sortBy(_.score).reverse.take(query.num)

    /*result.foreach(r=>{
      logger.info(s"itemID:${r._1},Array:${r._2}")
      r._2.foreach(println)
    })*/
    PredictedResult(result)

  }
}


