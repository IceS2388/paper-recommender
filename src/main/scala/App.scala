import recommender.Evaluation
import recommender.impl._

/**
  * Author:IceS
  * Date:2019-08-09 15:12:37
  * Description:
  * NONE
  */
object App {

  def main(args: Array[String]): Unit = {

    runBase()
    //runCluster()
    //runNCFCluster()
    //runRandomClusterForest()
    //runNCF()
    //runSAR()
    //runHot()

  }

  def runBase(): Unit = {
    //1.生成参数列表
    val args = List(
      //余弦相似度
      // numUserLikeMovies测试 结论：5最高，但是列表有不足10的情况
      //BaseParams(commonThreashold=2,numNearestUsers=5,numUserLikeMovies=5),
      //BaseParams(commonThreashold=2,numNearestUsers=5,numUserLikeMovies=10),
      //BaseParams(commonThreashold=2,numNearestUsers=5,numUserLikeMovies=20),
      //BaseParams(commonThreashold=2,numNearestUsers=5,numUserLikeMovies=40),
      //BaseParams(commonThreashold=2,numNearestUsers=5,numUserLikeMovies=80)
      //numNearestUsers测试 结论:5最高
      //BaseParams(commonThreashold=2,numNearestUsers=5,numUserLikeMovies=5),
      //BaseParams(commonThreashold=2,numNearestUsers=10,numUserLikeMovies=5),
      //BaseParams(commonThreashold=2,numNearestUsers=20,numUserLikeMovies=5),
      //BaseParams(commonThreashold=2,numNearestUsers=40,numUserLikeMovies=5),
      //BaseParams(commonThreashold=2,numNearestUsers=80,numUserLikeMovies=5)
      //commonThreashold 结论：5最佳
      //BaseParams(commonThreashold=2,numNearestUsers=5,numUserLikeMovies=5),
      //BaseParams(commonThreashold=5,numNearestUsers=5,numUserLikeMovies=5),
      //BaseParams(commonThreashold=10,numNearestUsers=5,numUserLikeMovies=5),
      //BaseParams(commonThreashold=20,numNearestUsers=5,numUserLikeMovies=5),
      //BaseParams(commonThreashold=40,numNearestUsers=5,numUserLikeMovies=5)
      //Cosine总结：最优参数 commonThreashold=5,numNearestUsers=5,numUserLikeMovies=5
      //准确率:0.1465,召回率:0.0969,f1:0.0934,时间:1(ms)

      //AdjustCosine
      //numUserLikeMovies测试 结论：5最高
      //BaseParams(method = "AdjustCosine",commonThreashold=5,numNearestUsers=5,numUserLikeMovies=5),
      //BaseParams(method = "AdjustCosine",commonThreashold=5,numNearestUsers=5,numUserLikeMovies=10),
      //BaseParams(method = "AdjustCosine",commonThreashold=5,numNearestUsers=5,numUserLikeMovies=20),
      //BaseParams(method = "AdjustCosine",commonThreashold=5,numNearestUsers=5,numUserLikeMovies=40),
      //BaseParams(method = "AdjustCosine",commonThreashold=5,numNearestUsers=5,numUserLikeMovies=80),
      //numNearestUsers测试 结论:5最高
      //BaseParams(method = "AdjustCosine",commonThreashold=5,numNearestUsers=5,numUserLikeMovies=5),
      //BaseParams(method = "AdjustCosine",commonThreashold=5,numNearestUsers=10,numUserLikeMovies=5),
      //BaseParams(method = "AdjustCosine",commonThreashold=5,numNearestUsers=20,numUserLikeMovies=5),
      //BaseParams(method = "AdjustCosine",commonThreashold=5,numNearestUsers=40,numUserLikeMovies=5),
      //BaseParams(method = "AdjustCosine",commonThreashold=5,numNearestUsers=80,numUserLikeMovies=5),
      //commonThreashold 结论：5最佳
      //BaseParams(method = "AdjustCosine",commonThreashold=2,numNearestUsers=5,numUserLikeMovies=5),
      //BaseParams(method = "AdjustCosine",commonThreashold=5,numNearestUsers=5,numUserLikeMovies=5),
      //BaseParams(method = "AdjustCosine",commonThreashold=10,numNearestUsers=5,numUserLikeMovies=5),
      //BaseParams(method = "AdjustCosine",commonThreashold=20,numNearestUsers=5,numUserLikeMovies=5),
      //BaseParams(method = "AdjustCosine",commonThreashold=40,numNearestUsers=5,numUserLikeMovies=5)
      //AdjustCosine总结：最优参数 commonThreashold=5,numNearestUsers=5,numUserLikeMovies=5
      //准确率:0.1493,召回率:0.0935,f1:0.0931,时间:0(ms)

      //Jaccard相似度
      //numUserLikeMovies测试 结论：5最高
      //BaseParams(method = "Jaccard",commonThreashold=2,numNearestUsers=5,numUserLikeMovies=5),
      //BaseParams(method = "Jaccard",commonThreashold=2,numNearestUsers=5,numUserLikeMovies=10),
      //BaseParams(method = "Jaccard",commonThreashold=2,numNearestUsers=5,numUserLikeMovies=20),
      //BaseParams(method = "Jaccard",commonThreashold=2,numNearestUsers=5,numUserLikeMovies=40),
      //BaseParams(method = "Jaccard",commonThreashold=2,numNearestUsers=5,numUserLikeMovies=80),
      //numNearestUsers测试 结论:5最高
      //BaseParams(method = "Jaccard",commonThreashold=2,numNearestUsers=5,numUserLikeMovies=5),
      //BaseParams(method = "Jaccard",commonThreashold=2,numNearestUsers=10,numUserLikeMovies=5),
      //BaseParams(method = "Jaccard",commonThreashold=2,numNearestUsers=20,numUserLikeMovies=5),
      //BaseParams(method = "Jaccard",commonThreashold=2,numNearestUsers=40,numUserLikeMovies=5),
      //BaseParams(method = "Jaccard",commonThreashold=2,numNearestUsers=80,numUserLikeMovies=5),
      //commonThreashold 结论：2最佳
      //BaseParams(method = "Jaccard",commonThreashold=2,numNearestUsers=5,numUserLikeMovies=5),
      //BaseParams(method = "Jaccard",commonThreashold=5,numNearestUsers=5,numUserLikeMovies=5),
      //BaseParams(method = "Jaccard",commonThreashold=10,numNearestUsers=5,numUserLikeMovies=5),
      //BaseParams(method = "Jaccard",commonThreashold=20,numNearestUsers=5,numUserLikeMovies=5),
      //BaseParams(method = "Jaccard",commonThreashold=40,numNearestUsers=5,numUserLikeMovies=5)
      //Jaccard总结：最优参数 commonThreashold=2,numNearestUsers=5,numUserLikeMovies=5
      //准确率:0.1514,召回率:0.1060,f1:0.1029,时间:1(ms)

      //JaccardMSD相似度
      //numUserLikeMovies测试 结论：5最高
      //BaseParams(method = "JaccardMSD",commonThreashold=2,numNearestUsers=5,numUserLikeMovies=5),
      //BaseParams(method = "JaccardMSD",commonThreashold=2,numNearestUsers=5,numUserLikeMovies=10),
      //BaseParams(method = "JaccardMSD",commonThreashold=2,numNearestUsers=5,numUserLikeMovies=20),
      //BaseParams(method = "JaccardMSD",commonThreashold=2,numNearestUsers=5,numUserLikeMovies=40),
      //BaseParams(method = "JaccardMSD",commonThreashold=2,numNearestUsers=5,numUserLikeMovies=80),
      //numNearestUsers测试 结论:5最高
      //BaseParams(method = "JaccardMSD",commonThreashold=2,numNearestUsers=5,numUserLikeMovies=5),
      //BaseParams(method = "JaccardMSD",commonThreashold=2,numNearestUsers=10,numUserLikeMovies=5),
      //BaseParams(method = "JaccardMSD",commonThreashold=2,numNearestUsers=20,numUserLikeMovies=5),
      //BaseParams(method = "JaccardMSD",commonThreashold=2,numNearestUsers=40,numUserLikeMovies=5),
      //BaseParams(method = "JaccardMSD",commonThreashold=2,numNearestUsers=80,numUserLikeMovies=5),
      //commonThreashold 结论：2最佳
      //BaseParams(method = "JaccardMSD",commonThreashold=2,numNearestUsers=5,numUserLikeMovies=5),
      //BaseParams(method = "JaccardMSD",commonThreashold=5,numNearestUsers=5,numUserLikeMovies=5),
      //BaseParams(method = "JaccardMSD",commonThreashold=10,numNearestUsers=5,numUserLikeMovies=5),
      //BaseParams(method = "JaccardMSD",commonThreashold=20,numNearestUsers=5,numUserLikeMovies=5),
      //BaseParams(method = "JaccardMSD",commonThreashold=40,numNearestUsers=5,numUserLikeMovies=5)
      //JaccardMSD总结：最优参数 commonThreashold=2,numNearestUsers=5,numUserLikeMovies=5
      //准确率:0.1392,召回率:0.0965,f1:0.0932,时间:0(ms)

      //Pearson
      // numUserLikeMovies测试 结论：5最高
      //BaseParams(method = "Pearson",commonThreashold=20,numNearestUsers=5,numUserLikeMovies=5),
      //BaseParams(method = "Pearson",commonThreashold=20,numNearestUsers=5,numUserLikeMovies=10),
      //BaseParams(method = "Pearson",commonThreashold=20,numNearestUsers=5,numUserLikeMovies=20),
      //BaseParams(method = "Pearson",commonThreashold=20,numNearestUsers=5,numUserLikeMovies=40),
      //BaseParams(method = "Pearson",commonThreashold=20,numNearestUsers=5,numUserLikeMovies=80),
      //numNearestUsers测试 结论:5最高
      //BaseParams(method = "Pearson",commonThreashold=20,numNearestUsers=5,numUserLikeMovies=5),
      //BaseParams(method = "Pearson",commonThreashold=20,numNearestUsers=10,numUserLikeMovies=5),
      //BaseParams(method = "Pearson",commonThreashold=20,numNearestUsers=20,numUserLikeMovies=5),
      //BaseParams(method = "Pearson",commonThreashold=20,numNearestUsers=40,numUserLikeMovies=5),
      //BaseParams(method = "Pearson",commonThreashold=20,numNearestUsers=80,numUserLikeMovies=5),
      //commonThreashold 结论：20最高
      //BaseParams(method = "Pearson",commonThreashold=2,numNearestUsers=5,numUserLikeMovies=5),
      //BaseParams(method = "Pearson",commonThreashold=5,numNearestUsers=5,numUserLikeMovies=5),
      //BaseParams(method = "Pearson",commonThreashold=10,numNearestUsers=5,numUserLikeMovies=5),
      //BaseParams(method = "Pearson",commonThreashold=20,numNearestUsers=5,numUserLikeMovies=5),
      //BaseParams(method = "Pearson",commonThreashold=40,numNearestUsers=5,numUserLikeMovies=5)
      //Pearson总结：最优参数 commonThreashold=20,numNearestUsers=5,numUserLikeMovies=5
      //准确率:0.0858,召回率:0.0318,f1:0.0413,时间:1(ms)

      //ImprovedPearson相似度
      // numUserLikeMovies测试 结论：5最高
      BaseParams(method = "ImprovedPearson",commonThreashold=2,numNearestUsers=5,numUserLikeMovies=5),
      BaseParams(method = "ImprovedPearson",commonThreashold=2,numNearestUsers=5,numUserLikeMovies=10),
      BaseParams(method = "ImprovedPearson",commonThreashold=2,numNearestUsers=5,numUserLikeMovies=20),
      BaseParams(method = "ImprovedPearson",commonThreashold=2,numNearestUsers=5,numUserLikeMovies=40),
      BaseParams(method = "ImprovedPearson",commonThreashold=2,numNearestUsers=5,numUserLikeMovies=80),
      //numNearestUsers测试 结论:5最高
      BaseParams(method = "ImprovedPearson",commonThreashold=2,numNearestUsers=5,numUserLikeMovies=5),
      BaseParams(method = "ImprovedPearson",commonThreashold=2,numNearestUsers=10,numUserLikeMovies=5),
      BaseParams(method = "ImprovedPearson",commonThreashold=2,numNearestUsers=20,numUserLikeMovies=5),
      BaseParams(method = "ImprovedPearson",commonThreashold=2,numNearestUsers=40,numUserLikeMovies=5),
      BaseParams(method = "ImprovedPearson",commonThreashold=2,numNearestUsers=80,numUserLikeMovies=5),
      //commonThreashold 结论：20最佳
      BaseParams(method = "ImprovedPearson",commonThreashold=2,numNearestUsers=5,numUserLikeMovies=5),
      BaseParams(method = "ImprovedPearson",commonThreashold=5,numNearestUsers=5,numUserLikeMovies=5),
      BaseParams(method = "ImprovedPearson",commonThreashold=10,numNearestUsers=5,numUserLikeMovies=5),
      BaseParams(method = "ImprovedPearson",commonThreashold=20,numNearestUsers=5,numUserLikeMovies=5),
      BaseParams(method = "ImprovedPearson",commonThreashold=40,numNearestUsers=5,numUserLikeMovies=5)
      //ImprovedPearson总结：最优参数 commonThreashold=20,numNearestUsers=5,numUserLikeMovies=5
      //准确率:0.0596,召回率:0.0202,f1:0.0260,时间:4(ms)
    )
    //结论：Cosine的准确率最高
    for (arg <- args) {
      val recommender = new BaseRecommender(arg)
      val eval = new Evaluation()
      eval.run(recommender)
    }
  }

  def runCluster(): Unit = {
    //1.生成参数列表
    val args = List(

      //BisectingKMeans
      // k: 4
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 2, maxIterations = 20),
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 3, maxIterations = 20),
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 20),
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 5, maxIterations = 20),
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 6, maxIterations = 20)
      //maxIterations:10
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 5),
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 10),
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 20),
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 30),
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 40)
      //
      ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 10),
      ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 10),
      ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 10),
      ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 10),
      ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 10)

    )
    for (arg <- args) {
      val recommender = new ClusterRecommender(arg)
      val eval = new Evaluation()
      eval.run(recommender)
    }
  }

  def runNCFCluster(): Unit = {
    val ap = NCFClusterParams()
    val recommender = new NCFClusterRecommender(ap)
    val eval = new Evaluation()
    eval.run(recommender)
  }

  def runRandomClusterForest(): Unit = {

    val ap = RandomForestClusterParams()
    val recommender = new RandomForestClusterRecommender(ap)
    val eval = new Evaluation()
    eval.run(recommender)
  }

  def runNCF(): Unit = {
    //单纯的NCF2分类
    val ap = NCFParams()
    val recommender = new NCFRecommender(ap)
    val eval = new Evaluation()
    eval.run(recommender)
  }

  def runSAR(): Unit = {
    val ap = SARParams()
    val recommender = new SARRecommender(ap)
    val eval = new Evaluation()
    eval.run(recommender)
  }

  def runHot(): Unit = {
    val ap = HotParams()
    val recommender = new HotRecommender(ap)
    val eval = new Evaluation()
    eval.run(recommender)
  }


}
