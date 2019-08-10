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
    runInprovedPearson()
    runPearson()
    runCosine()
  }
  def runInprovedPearson()={
    val ap=new BaseParams(method = "inprovedpearson")
    val recommender=new BaseRecommender(ap)
    val eval=new Evaluation()
    eval.run(recommender)
  }

  def runPearson()={
    val ap=new BaseParams(method = "pearson")
    val recommender=new BaseRecommender(ap)
    val eval=new Evaluation()
    eval.run(recommender)
  }

  def runCosine()={
    val ap=new BaseParams()
    val recommender=new BaseRecommender(ap)
    val eval=new Evaluation()
    eval.run(recommender)
  }
}
