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

    runCluster()
    //runInprovedPearson()
    //runPearson()
    //runCosine()
  }

  def runCluster():Unit={
    val ap=new ClusterParams()
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
