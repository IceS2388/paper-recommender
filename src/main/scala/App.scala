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

    //runBase()
    //runCluster()
    runNCFCluster()
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
      //准确率:0.1458,召回率:0.0960,f1:0.0926,时间:1(ms)

      //BaseParams(commonThreashold=2,numNearestUsers=5,numUserLikeMovies=10),
      //准确率:0.1412,召回率:0.0987,f1:0.0951,时间:0(ms)

      //BaseParams(commonThreashold=2,numNearestUsers=5,numUserLikeMovies=20),
      //准确率:0.1396,召回率:0.0974,f1:0.0940,时间:1(ms)

      //BaseParams(commonThreashold=2,numNearestUsers=5,numUserLikeMovies=40),
      //准确率:0.1392,召回率:0.0970,f1:0.0936,时间:0(ms)

      //BaseParams(commonThreashold=2,numNearestUsers=5,numUserLikeMovies=80)
      //准确率:0.1392,召回率:0.0970,f1:0.0936,时间:0(ms)

      //numNearestUsers测试 结论:5最高
      //BaseParams(commonThreashold=2,numNearestUsers=5,numUserLikeMovies=5),
      //准确率:0.1458,召回率:0.0960,f1:0.0926,时间:0(ms)

      //BaseParams(commonThreashold=2,numNearestUsers=10,numUserLikeMovies=5),
      //准确率:0.1322,召回率:0.0923,f1:0.0888,时间:1(ms)

      //BaseParams(commonThreashold=2,numNearestUsers=20,numUserLikeMovies=5),
      //准确率:0.1195,召回率:0.0824,f1:0.0795,时间:1(ms)

      //BaseParams(commonThreashold=2,numNearestUsers=40,numUserLikeMovies=5),
      //准确率:0.1036,召回率:0.0671,f1:0.0666,时间:1(ms)

      //BaseParams(commonThreashold=2,numNearestUsers=80,numUserLikeMovies=5)
      //准确率:0.0901,召回率:0.0565,f1:0.0569,时间:1(ms)

      //commonThreashold 结论：5最佳
      //BaseParams(commonThreashold=2,numNearestUsers=5,numUserLikeMovies=5),
      //准确率:0.1458,召回率:0.0960,f1:0.0926,时间:1(ms)

      //BaseParams(commonThreashold=5,numNearestUsers=5,numUserLikeMovies=5),
      //准确率:0.1465,召回率:0.0969,f1:0.0934,时间:1(ms)

      //BaseParams(commonThreashold=10,numNearestUsers=5,numUserLikeMovies=5),
      //准确率:0.1456,召回率:0.0888,f1:0.0890,时间:1(ms)

      //BaseParams(commonThreashold=20,numNearestUsers=5,numUserLikeMovies=5),
      //准确率:0.1180,召回率:0.0480,f1:0.0598,时间:0(ms)

      //BaseParams(commonThreashold=40,numNearestUsers=5,numUserLikeMovies=5)
      //准确率:0.0820,召回率:0.0214,f1:0.0322,时间:0(ms)

      //最优参数 commonThreashold=5,numNearestUsers=5,numUserLikeMovies=5
      //准确率:0.1465,召回率:0.0969,f1:0.0934,时间:1(ms)


      //AdjustCosine
      //numUserLikeMovies测试 结论：5最高
      //BaseParams(method = "AdjustCosine",commonThreashold=5,numNearestUsers=5,numUserLikeMovies=5),
      //准确率:0.1493,召回率:0.0935,f1:0.0931,时间:1(ms)

      //BaseParams(method = "AdjustCosine",commonThreashold=5,numNearestUsers=5,numUserLikeMovies=10),
      //准确率:0.1464,召回率:0.0994,f1:0.0967,时间:0(ms)

      //BaseParams(method = "AdjustCosine",commonThreashold=5,numNearestUsers=5,numUserLikeMovies=20),
      //准确率:0.1376,召回率:0.0922,f1:0.0903,时间:1(ms)

      //BaseParams(method = "AdjustCosine",commonThreashold=5,numNearestUsers=5,numUserLikeMovies=40),
      //准确率:0.1309,召回率:0.0891,f1:0.0864,时间:0(ms)

      //BaseParams(method = "AdjustCosine",commonThreashold=5,numNearestUsers=5,numUserLikeMovies=80),
      //准确率:0.1239,召回率:0.0862,f1:0.0829,时间:1(ms)

      //numNearestUsers测试 结论:5最高
      //BaseParams(method = "AdjustCosine",commonThreashold=5,numNearestUsers=5,numUserLikeMovies=5),
      //准确率:0.1493,召回率:0.0935,f1:0.0931,时间:0(ms)

      //BaseParams(method = "AdjustCosine",commonThreashold=5,numNearestUsers=10,numUserLikeMovies=5),
      //准确率:0.1405,召回率:0.0966,f1:0.0937,时间:1(ms)

      //BaseParams(method = "AdjustCosine",commonThreashold=5,numNearestUsers=20,numUserLikeMovies=5),
      //准确率:0.1291,召回率:0.0855,f1:0.0843,时间:1(ms)

      //BaseParams(method = "AdjustCosine",commonThreashold=5,numNearestUsers=40,numUserLikeMovies=5),
      //准确率:0.1150,召回率:0.0722,f1:0.0731,时间:0(ms)

      //BaseParams(method = "AdjustCosine",commonThreashold=5,numNearestUsers=80,numUserLikeMovies=5),
      //准确率:0.1010,召回率:0.0630,f1:0.0636,时间:1(ms)

      //commonThreashold 结论：5最佳
      //BaseParams(method = "AdjustCosine",commonThreashold=2,numNearestUsers=5,numUserLikeMovies=5),
      //准确率:0.1488,召回率:0.0923,f1:0.0924,时间:0(ms)

      //BaseParams(method = "AdjustCosine",commonThreashold=5,numNearestUsers=5,numUserLikeMovies=5),
      //准确率:0.1493,召回率:0.0935,f1:0.0931,时间:0(ms)

      //BaseParams(method = "AdjustCosine",commonThreashold=10,numNearestUsers=5,numUserLikeMovies=5),
      //准确率:0.1450,召回率:0.0880,f1:0.0887,时间:0(ms)

      //BaseParams(method = "AdjustCosine",commonThreashold=20,numNearestUsers=5,numUserLikeMovies=5),
      //准确率:0.1200,召回率:0.0480,f1:0.0603,时间:0(ms)

      //BaseParams(method = "AdjustCosine",commonThreashold=40,numNearestUsers=5,numUserLikeMovies=5)
      //准确率:0.0860,召回率:0.0220,f1:0.0331,时间:0(ms)

      //最优参数 commonThreashold=5,numNearestUsers=5,numUserLikeMovies=5
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
      //BaseParams(method = "ImprovedPearson",commonThreashold=2,numNearestUsers=5,numUserLikeMovies=5),
      //准确率:0.0464,召回率:0.0257,f1:0.0275,时间:2(ms)

      //BaseParams(method = "ImprovedPearson",commonThreashold=2,numNearestUsers=5,numUserLikeMovies=10),
      //准确率:0.0441,召回率:0.0246,f1:0.0262,时间:4(ms)

      //BaseParams(method = "ImprovedPearson",commonThreashold=2,numNearestUsers=5,numUserLikeMovies=20),
      //准确率:0.0437,召回率:0.0242,f1:0.0259,时间:2(ms)

      //BaseParams(method = "ImprovedPearson",commonThreashold=2,numNearestUsers=5,numUserLikeMovies=40),
      //准确率:0.0434,召回率:0.0238,f1:0.0255,时间:3(ms)

      //BaseParams(method = "ImprovedPearson",commonThreashold=2,numNearestUsers=5,numUserLikeMovies=80),
      //准确率:0.0434,召回率:0.0238,f1:0.0255,时间:3(ms)

      //numNearestUsers测试 结论:5最高
      //BaseParams(method = "ImprovedPearson",commonThreashold=2,numNearestUsers=5,numUserLikeMovies=5),
      //准确率:0.0464,召回率:0.0257,f1:0.0275,时间:1(ms)

      //BaseParams(method = "ImprovedPearson",commonThreashold=2,numNearestUsers=10,numUserLikeMovies=5),
      //准确率:0.0451,召回率:0.0245,f1:0.0264,时间:1(ms)

      //BaseParams(method = "ImprovedPearson",commonThreashold=2,numNearestUsers=20,numUserLikeMovies=5),
      //准确率:0.0417,召回率:0.0218,f1:0.0239,时间:2(ms)

      //BaseParams(method = "ImprovedPearson",commonThreashold=2,numNearestUsers=40,numUserLikeMovies=5),
      //准确率:0.0376,召回率:0.0189,f1:0.0211,时间:3(ms)

      //BaseParams(method = "ImprovedPearson",commonThreashold=2,numNearestUsers=80,numUserLikeMovies=5),
      //准确率:0.0331,召回率:0.0164,f1:0.0183,时间:5(ms)

      //commonThreashold 结论：10最佳
      //BaseParams(method = "ImprovedPearson",commonThreashold=2,numNearestUsers=5,numUserLikeMovies=5),
      //准确率:0.0464,召回率:0.0257,f1:0.0275,时间:2(ms)

      //BaseParams(method = "ImprovedPearson",commonThreashold=5,numNearestUsers=5,numUserLikeMovies=5),
      //准确率:0.0721,召回率:0.0424,f1:0.0434,时间:3(ms)

      //BaseParams(method = "ImprovedPearson",commonThreashold=10,numNearestUsers=5,numUserLikeMovies=5),
      //准确率:0.0876,召回率:0.0461,f1:0.0493,时间:3(ms)

      //BaseParams(method = "ImprovedPearson",commonThreashold=20,numNearestUsers=5,numUserLikeMovies=5),
      //准确率:0.0874,召回率:0.0323,f1:0.0418,时间:3(ms)

      //BaseParams(method = "ImprovedPearson",commonThreashold=40,numNearestUsers=5,numUserLikeMovies=5)
      //准确率:0.0744,召回率:0.0187,f1:0.0284,时间:2(ms)

      //最优参数 commonThreashold=10,numNearestUsers=5,numUserLikeMovies=5
      //准确率:0.0876,召回率:0.0461,f1:0.0493,时间:3(ms)
    )
    //结论：Jaccard的准确率最高
    //Jaccard总结：最优参数 commonThreashold=2,numNearestUsers=5,numUserLikeMovies=5
    //准确率:0.1514,召回率:0.1060,f1:0.1029,时间:1(ms)

    for (arg <- args) {
      val recommender = new BaseRecommender(arg)
      val eval = new Evaluation()
      eval.run(recommender)
    }
  }

  def runCluster(): Unit = {
    //1.生成参数列表
    val args = List(

      /***------------------BisectingKMeans------------------***/
      //BisectingKMeans
      // k: 2 变化波动不太大
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 2, maxIterations = 10),
      //准确率:0.0703,召回率:0.0363,f1:0.0395,时间:144(ms)
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 3, maxIterations = 10),
      //准确率:0.0697,召回率:0.0362,f1:0.0393,时间:93(ms)
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 10),
      //准确率:0.0697,召回率:0.0365,f1:0.0394,时间:75(ms)
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 5, maxIterations = 10),
      //准确率:0.0701,召回率:0.0366,f1:0.0395,时间:66(ms)
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 6, maxIterations = 10),
      //准确率:0.0701,召回率:0.0367,f1:0.0396,时间:69(ms)
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 7, maxIterations = 10),
      //准确率:0.0702,召回率:0.0367,f1:0.0396,时间:35(ms)
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 8, maxIterations = 10),
      //准确率:0.0698,召回率:0.0367,f1:0.0394,时间:40(ms)
      //maxIterations:5 超过20后无变化
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 5),
      //准确率:0.0706,召回率:0.0367,f1:0.0399,时间:69(ms)
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 10),
      //准确率:0.0697,召回率:0.0365,f1:0.0394,时间:66(ms)
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 20),
      //准确率:0.0698,召回率:0.0364,f1:0.0395,时间:79(ms)
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 30),
      //准确率:0.0698,召回率:0.0364,f1:0.0395,时间:75(ms)
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 40),
      //准确率:0.0698,召回率:0.0364,f1:0.0395,时间:72(ms)
      //numNearestUsers:240
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 10,numNearestUsers=100),
      //准确率:0.1353,召回率:0.0771,f1:0.0813,时间:82(ms)
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 10,numNearestUsers=120),
      //准确率:0.1385,召回率:0.0802,f1:0.0840,时间:88(ms)
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 10,numNearestUsers=140),
      //准确率:0.1409,召回率:0.0823,f1:0.0860,时间:120(ms)
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 10,numNearestUsers=160),
      //准确率:0.1413,召回率:0.0827,f1:0.0864,时间:51(ms)
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 10,numNearestUsers=180),
      //准确率:0.1429,召回率:0.0837,f1:0.0875,时间:76(ms)
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 10,numNearestUsers=200),
      //准确率:0.1445,召回率:0.0856,f1:0.0891,时间:64(ms)
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 10,numNearestUsers=220),
      //准确率:0.1451,召回率:0.0868,f1:0.0899,时间:67(ms)
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 10,numNearestUsers=240),
      //准确率:0.1460,召回率:0.0877,f1:0.0907,时间:68(ms)
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 10,numNearestUsers=260)
      //准确率:0.1452,召回率:0.0878,f1:0.0906,时间:63(ms)
      //numUserLikeMovies:240
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 10,numNearestUsers=240,numUserLikeMovies=5),
      //准确率:0.1460,召回率:0.0877,f1:0.0907,时间:70(ms)
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 10,numNearestUsers=240,numUserLikeMovies=10),
      //准确率:0.1545,召回率:0.0944,f1:0.0967,时间:67(ms)
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 10,numNearestUsers=240,numUserLikeMovies=20),
      //准确率:0.1622,召回率:0.0978,f1:0.1008,时间:76(ms)
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 10,numNearestUsers=240,numUserLikeMovies=40),
      //准确率:0.1684,召回率:0.0984,f1:0.1027,时间:71(ms)
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 10,numNearestUsers=240,numUserLikeMovies=80),
      //准确率:0.1771,召回率:0.0991,f1:0.1058,时间:74(ms)
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 10,numNearestUsers=240,numUserLikeMovies=120),
      //准确率:0.1809,召回率:0.0985,f1:0.1066,时间:94(ms)
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 10,numNearestUsers=240,numUserLikeMovies=160),
      //准确率:0.1825,召回率:0.0997,f1:0.1077,时间:84(ms)
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 10,numNearestUsers=240,numUserLikeMovies=200),
      //准确率:0.1829,召回率:0.1001,f1:0.1081,时间:86(ms)
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 10,numNearestUsers=240,numUserLikeMovies=240),
      //准确率:0.1833,召回率:0.1002,f1:0.1084,时间:94(ms)
      //ClusterParams(clusterMethod = "BisectingKMeans", k = 4, maxIterations = 10,numNearestUsers=240,numUserLikeMovies=280)
      //准确率:0.1833,召回率:0.1005,f1:0.1086,时间:95(ms)

      //最优参数：K:4,maxIterations:10,numNearestUsers:240,numUserLikeMovies=240
      //最好结果：准确率:0.1833,召回率:0.1002,f1:0.1084,时间:94(ms)

      /***------------------K-means------------------***/
      //maxIterations:10
      //ClusterParams(clusterMethod = "K-means", k = 4, maxIterations = 5, numNearestUsers = 240, numUserLikeMovies = 240),
      //准确率:0.1839,召回率:0.1012,f1:0.1089,时间:102(ms)

      //ClusterParams(clusterMethod = "K-means", k = 4, maxIterations = 10, numNearestUsers = 240, numUserLikeMovies = 240),
      //准确率:0.1849,召回率:0.1014,f1:0.1092,时间:112(ms)

      //ClusterParams(clusterMethod = "K-means", k = 4, maxIterations = 20, numNearestUsers = 240, numUserLikeMovies = 240),
      //准确率:0.1845,召回率:0.1000,f1:0.1085,时间:121(ms)

      //ClusterParams(clusterMethod = "K-means", k = 4, maxIterations = 40, numNearestUsers = 240, numUserLikeMovies = 240),
      //准确率:0.1844,召回率:0.1000,f1:0.1085,时间:110(ms)

      //ClusterParams(clusterMethod = "K-means", k = 4, maxIterations = 80, numNearestUsers = 240, numUserLikeMovies = 240)
      //准确率:0.1846,召回率:0.1008,f1:0.1088,时间:114(ms)

      //k:4
      //ClusterParams(clusterMethod = "K-means", k = 3, maxIterations = 10, numNearestUsers = 240, numUserLikeMovies = 240),
      //准确率:0.1840,召回率:0.0996,f1:0.1082,时间:94(ms)

      //ClusterParams(clusterMethod = "K-means", k = 4, maxIterations = 10, numNearestUsers = 240, numUserLikeMovies = 240),
      //准确率:0.1846,召回率:0.1007,f1:0.1087,时间:79(ms)

      //ClusterParams(clusterMethod = "K-means", k = 5, maxIterations = 10, numNearestUsers = 240, numUserLikeMovies = 240),
      //准确率:0.1818,召回率:0.1019,f1:0.1086,时间:101(ms)

      //ClusterParams(clusterMethod = "K-means", k = 6, maxIterations = 10, numNearestUsers = 240, numUserLikeMovies = 240),
      //准确率:0.1842,召回率:0.1044,f1:0.1109,时间:69(ms)

      //ClusterParams(clusterMethod = "K-means", k = 7, maxIterations = 10, numNearestUsers = 240, numUserLikeMovies = 240)
      //准确率:0.1824,召回率:0.1028,f1:0.1094,时间:71(ms)

      //K-means最优参数：k = 4, maxIterations = 10, numNearestUsers = 240, numUserLikeMovies = 240
      //准确率:0.1849,召回率:0.1014,f1:0.1092,时间:112(ms)

      /***------------------GaussianMixture------------------***/
      //GaussianMixture
      //k:2
      //ClusterParams(clusterMethod = "GaussianMixture", k = 2, maxIterations = 10, numNearestUsers = 240, numUserLikeMovies = 240),
      //准确率:0.1861,召回率:0.0972,f1:0.1071,时间:269(ms)
      //ClusterParams(clusterMethod = "GaussianMixture", k = 3, maxIterations = 10, numNearestUsers = 240, numUserLikeMovies = 240),
      //准确率:0.1861,召回率:0.0971,f1:0.1070,时间:302(ms)
      //ClusterParams(clusterMethod = "GaussianMixture", k = 4, maxIterations = 10, numNearestUsers = 240, numUserLikeMovies = 240),
      //准确率:0.1858,召回率:0.0970,f1:0.1069,时间:186(ms)
      //ClusterParams(clusterMethod = "GaussianMixture", k = 5, maxIterations = 10, numNearestUsers = 240, numUserLikeMovies = 240),
      //准确率:0.1858,召回率:0.0966,f1:0.1066,时间:187(ms)
      //ClusterParams(clusterMethod = "GaussianMixture", k = 6, maxIterations = 10, numNearestUsers = 240, numUserLikeMovies = 240),
      //准确率:0.1853,召回率:0.0965,f1:0.1065,时间:153(ms)
      //ClusterParams(clusterMethod = "GaussianMixture", k = 7, maxIterations = 10, numNearestUsers = 240, numUserLikeMovies = 240),
      //准确率:0.1853,召回率:0.0963,f1:0.1065,时间:155(ms)
      //maxIterations:10
      //ClusterParams(clusterMethod = "GaussianMixture", k = 2, maxIterations = 10, numNearestUsers = 240, numUserLikeMovies = 240),
      //准确率:0.1862,召回率:0.0973,f1:0.1071,时间:220(ms)
      //ClusterParams(clusterMethod = "GaussianMixture", k = 2, maxIterations = 20, numNearestUsers = 240, numUserLikeMovies = 240),
      //准确率:0.1861,召回率:0.0972,f1:0.1071,时间:210(ms)
      //ClusterParams(clusterMethod = "GaussianMixture", k = 2, maxIterations = 30, numNearestUsers = 240, numUserLikeMovies = 240),
      //准确率:0.1862,召回率:0.0971,f1:0.1071,时间:177(ms)
      //ClusterParams(clusterMethod = "GaussianMixture", k = 2, maxIterations = 40, numNearestUsers = 240, numUserLikeMovies = 240),
      //准确率:0.1861,召回率:0.0971,f1:0.1070,时间:188(ms)
      //ClusterParams(clusterMethod = "GaussianMixture", k = 2, maxIterations = 50, numNearestUsers = 240, numUserLikeMovies = 240)
      //准确率:0.1861,召回率:0.0973,f1:0.1071,时间:206(ms)

      //GaussianMixture:簇心数量:2,maxIterations:10,相似度方法:Cosine,numNearestUsers:240,numUserLikeMovies:240
      //准确率:0.1862,召回率:0.0973,f1:0.1071,时间:220(ms)


    )
    for (arg <- args) {
      val recommender = new ClusterRecommender(arg)
      val eval = new Evaluation()
      eval.run(recommender)
    }
  }

  def runNCFCluster(): Unit = {

    val args=List(
      NCFClusterParams(maxIterations=10,numNearestUsers=240,numUserLikeMovies=240)
    )
    for (elem <- args) {
      val recommender = new NCFClusterRecommender(elem)
      val eval = new Evaluation()
      eval.run(recommender)
    }

  }

  def runRandomClusterForest(): Unit = {

    //BisectingKMeans K:4,maxIterations:10,numNearestUsers:240,numUserLikeMovies=240
    //cosine:commonThreashold=5,numNearestUsers=5,numUserLikeMovies=5
    val args = List(
      RandomForestClusterParams(k = 4,numNearestUsers = 240,numUserLikeMovies=240,maxIterationsCluster=10)//,
      //RandomForestClusterParams()
    )
    for (elem <- args) {
      val recommender = new RandomForestClusterRecommender(elem)
      val eval = new Evaluation()
      eval.run(recommender)
    }

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
