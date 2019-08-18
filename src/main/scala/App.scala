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
    //runNCFCluster()
    //runRandomForest()
    runNCF()
    //runSAR()
    //runJaccard()
    //runCosine()
    //runPearson()
    //runInprovedPearson()
    //runHot()
    //runClusterPearson()
    //runClusterCosine()
    //runClusterImprovedPearson()
  }

  def runNCFCluster():Unit={
    val ap=new NCFClusterParams()
    val recommender=new NCFClusterRecommender(ap)
    val eval=new Evaluation()
    eval.run(recommender)
  }

  def runRandomForest():Unit={
    val ap=new RandomForestParams()
    val recommender=new RandomForestRecommender(ap)
    val eval=new Evaluation()
    eval.run(recommender)
  }

  def runNCF():Unit={
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
  def runClusterImprovedPearson():Unit={
    val ap=new ClusterParams(method = "ImprovedPearson")
    val recommender=new ClusterRecommender(ap)
    val eval=new Evaluation()
    eval.run(recommender)
  }
  def runClusterCosine():Unit={
    val ap=new ClusterParams(method = "Cosine")
    val recommender=new ClusterRecommender(ap)
    val eval=new Evaluation()
    eval.run(recommender)
  }

  def runClusterPearson():Unit={
    val ap=new ClusterParams(method = "Pearson")
    val recommender=new ClusterRecommender(ap)
    val eval=new Evaluation()
    eval.run(recommender)
  }

  def runInprovedPearson():Unit={
    val ap=new BaseParams(method = "InprovedPearson")
    val recommender=new BaseRecommender(ap)
    val eval=new Evaluation()
    eval.run(recommender)
  }

  def runJaccard():Unit={
    val ap=new BaseParams(method = "Jaccard")
    val recommender=new BaseRecommender(ap)
    val eval=new Evaluation()
    eval.run(recommender)
  }

  def runPearson():Unit={
    val ap=new BaseParams(method = "Pearson")
    val recommender=new BaseRecommender(ap)
    val eval=new Evaluation()
    eval.run(recommender)
  }

  def runCosine():Unit={
    val ap=new BaseParams()
    val recommender=new BaseRecommender(ap)
    val eval=new Evaluation()
    eval.run(recommender)
  }
}
