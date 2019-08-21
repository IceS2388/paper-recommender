import org.slf4j.{Logger, LoggerFactory}

/**
  * Author:IceS
  * Date:2019-08-09 15:12:37
  * Description:
  * NONE
  */
object App {
  @transient private lazy val logger: Logger =LoggerFactory.getLogger(this.getClass)
  def main(args: Array[String]): Unit = {

    runBase()
    //runNCFCluster()
    //runRandomClusterForest()
    //runNCF()
    //runSAR()

    //runHot()
    //runClusterPearson()
    //runClusterCosine()
    //runClusterImprovedPearson()
  }

  def runBase():Unit={
    //1.生成参数列表
    val args=List(
      //余弦相似度
      // numUserLikeMovies测试
      //new BaseParams(commonThreashold=2,numNearestUsers=5,numUserLikeMovies=5),
      //new BaseParams(commonThreashold=2,numNearestUsers=5,numUserLikeMovies=10),
      //new BaseParams(commonThreashold=2,numNearestUsers=5,numUserLikeMovies=20),
      //new BaseParams(commonThreashold=2,numNearestUsers=5,numUserLikeMovies=40),
      //new BaseParams(commonThreashold=2,numNearestUsers=5,numUserLikeMovies=80),
      //numNearestUsers测试
      new BaseParams(commonThreashold=2,numNearestUsers=5,numUserLikeMovies=5),
      new BaseParams(commonThreashold=2,numNearestUsers=10,numUserLikeMovies=5),
      new BaseParams(commonThreashold=2,numNearestUsers=20,numUserLikeMovies=5),
      new BaseParams(commonThreashold=2,numNearestUsers=40,numUserLikeMovies=5),
      new BaseParams(commonThreashold=2,numNearestUsers=80,numUserLikeMovies=5)

    )
    for( arg <- args){
      val recommender=new BaseRecommender(arg)
      val eval=new Evaluation()
      eval.run(recommender)
    }
  }

  def runCluster():Unit={
    //1.生成参数列表
    val args=List(
      //余弦相似度
      // numUserLikeMovies测试
      new ClusterParams(numNearestUsers=5,numUserLikeMovies=5),
      new ClusterParams(numNearestUsers=5,numUserLikeMovies=10),
      new ClusterParams(numNearestUsers=5,numUserLikeMovies=20),
      new ClusterParams(numNearestUsers=5,numUserLikeMovies=40),
      new ClusterParams(numNearestUsers=5,numUserLikeMovies=80)

    )
    for( arg <- args){
      val recommender=new ClusterRecommender(arg)
      val eval=new Evaluation()
      eval.run(recommender)
    }
  }

  def runNCFCluster():Unit={
    val ap=new NCFClusterParams()
    val recommender=new NCFClusterRecommender(ap)
    val eval=new Evaluation()
    eval.run(recommender)
  }

  def runRandomClusterForest():Unit={

    val ap=new RandomForestClusterParams()
    val recommender=new RandomForestClusterRecommender(ap)
    val eval=new Evaluation()
    eval.run(recommender)
  }

  def runNCF():Unit={
    //单纯的NCF2分类
    val ap=new NCFParams()
    val recommender=new NCFRecommender(ap)
    val eval=new Evaluation()
    eval.run(recommender)
  }

  def runSAR():Unit={
    val ap=new SARParams()
    val recommender=new SARRecommender(ap)
    val eval=new Evaluation()
    eval.run(recommender)
  }

  def runHot():Unit={
    val ap=new HotParams()
    val recommender=new HotRecommender(ap)
    val eval=new Evaluation()
    eval.run(recommender)
  }



}
