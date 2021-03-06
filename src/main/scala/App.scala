import recommender.Evaluation
import recommender.impl._

/**
  * Author:IceS
  * Date:2019-08-09 15:12:37
  * Description:
  * 运行很简单，调参有点复杂。
  */
object App {

  def main(args: Array[String]): Unit = {

    runHot()
    runBase()
    runCluster()
    runSARCluster()

    //runKerasCluster()
    //runRandomClusterForest()
    //runNCFCluster()
    //runNCF()
    //runSAR()


  }

  def runHot(): Unit = {
    val ap = HotParams()
    val recommender = new HotRecommender(ap)
    val eval = new Evaluation()
    eval.run(recommender)
  }
  def runBase(): Unit = {
    //1.生成参数列表
    val args = List(
      /**
        * method:Cosine,T:2,K:5,L:5
        * */
      // L测试 结论：5最高，但是列表有不足10的情况
      //BasicParams(T=2,K=5,L=5),
      //准确率:0.1458,召回率:0.0960,f1:0.0926,时间:1(ms)
      /**
        *T=2,K=5,L=10
        * */
      //BasicParams(T=2,K=5,L=10),
      //准确率:0.1412,召回率:0.0987,f1:0.0951,时间:0(ms)
      /**
        *T=2,K=5,L=20
        * */
      //BasicParams(T=2,K=5,L=20),
      //准确率:0.1396,召回率:0.0974,f1:0.0940,时间:1(ms)
      /**
        *T=2,K=5,L=40
        * */
      //BasicParams(T=2,K=5,L=40),
      //准确率:0.1392,召回率:0.0970,f1:0.0936,时间:0(ms)
      /**
        *T=2,K=5,L=80
        * */
      //BasicParams(T=2,K=5,L=80)
      //准确率:0.1392,召回率:0.0970,f1:0.0936,时间:0(ms)


      /**
        *T=2,K=5,L=5
        * */
      //K测试 结论:5最高
      //BasicParams(T=2,K=5,L=5),
      //准确率:0.1458,召回率:0.0960,f1:0.0926,时间:0(ms)
      /**
        *T=2,K=10,L=5
        * */
      //BasicParams(T=2,K=10,L=5),
      //准确率:0.1322,召回率:0.0923,f1:0.0888,时间:1(ms)
      /**
        *T=2,K=20,L=5
        * */
      //BasicParams(T=2,K=20,L=5),
      //准确率:0.1195,召回率:0.0824,f1:0.0795,时间:1(ms)
      /**
        *T=2,K=40,L=5
        * */
      //BasicParams(T=2,K=40,L=5),
      //准确率:0.1036,召回率:0.0671,f1:0.0666,时间:1(ms)
      /**
        *T=2,K=80,L=5
        * */
      //BasicParams(T=2,K=80,L=5)
      //准确率:0.0901,召回率:0.0565,f1:0.0569,时间:1(ms)

      /**
        *T=2,K=5,L=5
        * */
      //T 结论：5最佳
      BasicParams(T=2,K=5,L=5),
      //准确率:0.1458,召回率:0.0960,f1:0.0926,时间:1(ms)
      /**
        *T=5,K=5,L=5
        * */
      //BasicParams(T=5,K=5,L=5),
      //准确率:0.1465,召回率:0.0969,f1:0.0934,时间:1(ms)
      /**
        *T=10,K=5,L=5
        * */
      //BasicParams(T=10,K=5,L=5),
      //准确率:0.1456,召回率:0.0888,f1:0.0890,时间:1(ms)
      /**
        *T=20,K=5,L=5
        * */
      //BasicParams(T=20,K=5,L=5),
      //准确率:0.1180,召回率:0.0480,f1:0.0598,时间:0(ms)
      /**
        *T=40,K=5,L=5
        * */
      //BasicParams(T=40,K=5,L=5)
      //准确率:0.0820,召回率:0.0214,f1:0.0322,时间:0(ms)


      /**
        *
        * 最优参数 T = 5, K = 5, L = 5
        * 准确率:0.1465,召回率:0.0969,f1:0.0934,时间:1(ms)
        * */
      /**
        *method = "AdjustCosine",T=5,K=5,L=5
        * */
      //AdjustCosine
      //L测试 结论：5最高
      //BasicParams(method = "AdjustCosine",T=5,K=5,L=5),
      //准确率:0.1493,召回率:0.0935,f1:0.0931,时间:1(ms)
      /**
        *method = "AdjustCosine",T=5,K=5,L=10
        * */
      //BasicParams(method = "AdjustCosine",T=5,K=5,L=10),
      //准确率:0.1464,召回率:0.0994,f1:0.0967,时间:0(ms)
      /**
        *method = "AdjustCosine",T=5,K=5,L=20
        * */
      //BasicParams(method = "AdjustCosine",T=5,K=5,L=20),
      //准确率:0.1376,召回率:0.0922,f1:0.0903,时间:1(ms)
      /**
        *method = "AdjustCosine",T=5,K=5,L=40
        * */
      //BasicParams(method = "AdjustCosine",T=5,K=5,L=40),
      //准确率:0.1309,召回率:0.0891,f1:0.0864,时间:0(ms)
      /**
        *method = "AdjustCosine",T=5,K=5,L=80
        * */
      //BasicParams(method = "AdjustCosine",T=5,K=5,L=80),
      //准确率:0.1239,召回率:0.0862,f1:0.0829,时间:1(ms)
      /**
        *method = "AdjustCosine",T=5,K=5,L=5
        * */
      //K测试 结论:5最高
      //BasicParams(method = "AdjustCosine",T=5,K=5,L=5),
      //准确率:0.1493,召回率:0.0935,f1:0.0931,时间:0(ms)
      /**
        *method = "AdjustCosine",T=5,K=10,L=5
        * */
      //BasicParams(method = "AdjustCosine",T=5,K=10,L=5),
      //准确率:0.1405,召回率:0.0966,f1:0.0937,时间:1(ms)
      /**
        *method = "AdjustCosine",T=5,K=20,L=5
        * */
      //BasicParams(method = "AdjustCosine",T=5,K=20,L=5),
      //准确率:0.1291,召回率:0.0855,f1:0.0843,时间:1(ms)
      /**
        *method = "AdjustCosine",T=5,K=40,L=5
        * */
      //BasicParams(method = "AdjustCosine",T=5,K=40,L=5),
      //准确率:0.1150,召回率:0.0722,f1:0.0731,时间:0(ms)
      /**
        *method = "AdjustCosine",T=5,K=80,L=5
        * */
      //BasicParams(method = "AdjustCosine",T=5,K=80,L=5),
      //准确率:0.1010,召回率:0.0630,f1:0.0636,时间:1(ms)
      /**
        *method = "AdjustCosine",T=2,K=5,L=5
        * */
      //T 结论：5最佳
      //BasicParams(method = "AdjustCosine",T=2,K=5,L=5),
      //准确率:0.1488,召回率:0.0923,f1:0.0924,时间:0(ms)
      /**
        *method = "AdjustCosine",T=5,K=5,L=5
        * */
      //BasicParams(method = "AdjustCosine",T=5,K=5,L=5),
      //准确率:0.1493,召回率:0.0935,f1:0.0931,时间:0(ms)
      /**
        *method = "AdjustCosine",T=10,K=5,L=5
        * */
      //BasicParams(method = "AdjustCosine",T=10,K=5,L=5),
      //准确率:0.1450,召回率:0.0880,f1:0.0887,时间:0(ms)
      /**
        *method = "AdjustCosine",T=20,K=5,L=5
        * */
      //BasicParams(method = "AdjustCosine",T=20,K=5,L=5),
      //准确率:0.1200,召回率:0.0480,f1:0.0603,时间:0(ms)
      /**
        *method = "AdjustCosine",T=40,K=5,L=5
        * */
      //BasicParams(method = "AdjustCosine",T=40,K=5,L=5)
      //准确率:0.0860,召回率:0.0220,f1:0.0331,时间:0(ms)
      /**
        *最优参数 T=5,K=5,L=5
        * 准确率:0.1493,召回率:0.0935,f1:0.0931,时间:0(ms)
        * */
      BasicParams(method = "AdjustCosine",T=5,K=5,L=5),



      //Jaccard相似度
      //L测试 结论：5最高
      /**
        *method = "Jaccard",T=2,K=5,L=5
        * */
      //BasicParams(method = "Jaccard",T=2,K=5,L=5),
      /**准确率:0.1514,召回率:0.1060,f1:0.1029,时间:1(ms) */

      //BasicParams(method = "Jaccard",T=2,K=5,L=10),
      /**准确率:0.1437,召回率:0.1045,f1:0.0994,时间:1(ms)  */

      //BasicParams(method = "Jaccard",T=2,K=5,L=20),
      /**准确率:0.1383,召回率:0.0996,f1:0.0948,时间:1(ms)   */

      //BasicParams(method = "Jaccard",T=2,K=5,L=40),
      /**准确率:0.1321,召回率:0.0967,f1:0.0911,时间:1(ms) */

      //BasicParams(method = "Jaccard",T=2,K=5,L=80),
      /**准确率:0.1253,召回率:0.0949,f1:0.0885,时间:1(ms) */

      //K测试 结论:5最高
      // BasicParams(method = "Jaccard",T=2,K=5,L=5),
      /**准确率:0.1514,召回率:0.1060,f1:0.1029,时间:1(ms)  */
      //BasicParams(method = "Jaccard",T=2,K=10,L=5),
      /**准确率:0.1404,召回率:0.1001,f1:0.0966,时间:1(ms) */
      //BasicParams(method = "Jaccard",T=2,K=20,L=5),
      /**准确率:0.1273,召回率:0.0871,f1:0.0853,时间:1(ms)  */
      //BasicParams(method = "Jaccard",T=2,K=40,L=5),
      /**准确率:0.1112,召回率:0.0719,f1:0.0722,时间:1(ms) */
      //BasicParams(method = "Jaccard",T=2,K=80,L=5),
      /**准确率:0.0952,召回率:0.0602,f1:0.0605,时间:2(ms)  */

      //T 结论：2最佳
      //BasicParams(method = "Jaccard",T=2,K=5,L=5),
      /**准确率:0.1514,召回率:0.1060,f1:0.1029,时间:1(ms) */
      //BasicParams(method = "Jaccard",T=5,K=5,L=5),
      /**准确率:0.1513,召回率:0.1060,f1:0.1028,时间:0(ms) */
      //BasicParams(method = "Jaccard",T=10,K=5,L=5),
      /**准确率:0.1449,召回率:0.0957,f1:0.0952,时间:0(ms) */
      //BasicParams(method = "Jaccard",T=20,K=5,L=5),
      /**准确率:0.1150,召回率:0.0494,f1:0.0612,时间:0(ms) */
      //BasicParams(method = "Jaccard",T=40,K=5,L=5)
      /**准确率:0.0840,召回率:0.0226,f1:0.0340,时间:0(ms) */

      //Jaccard总结：最优参数 T=2,K=5,L=5
      //准确率:0.1514,召回率:0.1060,f1:0.1029,时间:1(ms)
      BasicParams(method = "Jaccard",T=2,K=5,L=5),

      //JaccardMSD相似度
      //L测试 结论：5最高
      //BasicParams(method = "JaccardMSD",T=2,K=5,L=5),
      /**准确率:0.1392,召回率:0.0965,f1:0.0932,时间:1(ms) */
      //BasicParams(method = "JaccardMSD",T=2,K=5,L=10),
      /**准确率:0.1328,召回率:0.0934,f1:0.0897,时间:0(ms) */
      //BasicParams(method = "JaccardMSD",T=2,K=5,L=20),
      /**准确率:0.1285,召回率:0.0887,f1:0.0861,时间:0(ms) */
      //BasicParams(method = "JaccardMSD",T=2,K=5,L=40),
      /**准确率:0.1220,召回率:0.0850,f1:0.0820,时间:0(ms)  */
      //BasicParams(method = "JaccardMSD",T=2,K=5,L=80),
      /**准确率:0.1143,召回率:0.0825,f1:0.0786,时间:0(ms) */
      //K测试 结论:5最高
      //BasicParams(method = "JaccardMSD",T=2,K=5,L=5),
      /**准确率:0.1392,召回率:0.0965,f1:0.0932,时间:0(ms) */
      //BasicParams(method = "JaccardMSD",T=2,K=10,L=5),
      /**准确率:0.1290,召回率:0.0888,f1:0.0863,时间:0(ms) */
      //BasicParams(method = "JaccardMSD",T=2,K=20,L=5),
      /**准确率:0.1182,召回率:0.0790,f1:0.0776,时间:0(ms) */
      //BasicParams(method = "JaccardMSD",T=2,K=40,L=5),
      /**准确率:0.1045,召回率:0.0674,f1:0.0671,时间:0(ms) */
      //BasicParams(method = "JaccardMSD",T=2,K=80,L=5),
      /**准确率:0.0903,召回率:0.0580,f1:0.0574,时间:1(ms) */
      //T 结论：2最佳
      //BasicParams(method = "JaccardMSD",T=2,K=5,L=5),
      /**准确率:0.1392,召回率:0.0965,f1:0.0932,时间:0(ms) */
      //BasicParams(method = "JaccardMSD",T=5,K=5,L=5),
      /**准确率:0.1389,召回率:0.0959,f1:0.0928,时间:0(ms) */
      //BasicParams(method = "JaccardMSD",T=10,K=5,L=5),
      /**准确率:0.1320,召回率:0.0836,f1:0.0844,时间:0(ms) */
      //BasicParams(method = "JaccardMSD",T=20,K=5,L=5),
      /**准确率:0.1067,召回率:0.0435,f1:0.0552,时间:0(ms) */
      //BasicParams(method = "JaccardMSD",T=40,K=5,L=5)
      /**准确率:0.0800,召回率:0.0215,f1:0.0323,时间:0(ms) */

      //JaccardMSD总结：最优参数 T=2,K=5,L=5
      //准确率:0.1392,召回率:0.0965,f1:0.0932,时间:0(ms)
      BasicParams(method = "JaccardMSD",T=2,K=5,L=5),

      //Pearson
      // L测试 结论：5最高
      //BasicParams(method = "Pearson",T=20,K=5,L=5),
      /**准确率:0.0858,召回率:0.0318,f1:0.0413,时间:1(ms) */
      //BasicParams(method = "Pearson",T=20,K=5,L=10),
      /**准确率:0.0818,召回率:0.0315,f1:0.0403,时间:2(ms) */
      //BasicParams(method = "Pearson",T=20,K=5,L=20),
      /**准确率:0.0771,召回率:0.0309,f1:0.0386,时间:2(ms) */
      //BasicParams(method = "Pearson",T=20,K=5,L=40),
      /**准确率:0.0742,召回率:0.0297,f1:0.0371,时间:1(ms) */
      //BasicParams(method = "Pearson",T=20,K=5,L=80),
      /**准确率:0.0707,召回率:0.0275,f1:0.0348,时间:1(ms) */
      //K测试 结论:5最高
      //BasicParams(method = "Pearson",T=20,K=5,L=5),
      /**准确率:0.0858,召回率:0.0318,f1:0.0413,时间:0(ms) */
      //BasicParams(method = "Pearson",T=20,K=10,L=5),
      /**准确率:0.0816,召回率:0.0305,f1:0.0398,时间:0(ms) */
      //BasicParams(method = "Pearson",T=20,K=20,L=5),
      /**准确率:0.0760,召回率:0.0286,f1:0.0372,时间:1(ms) */
      //BasicParams(method = "Pearson",T=20,K=40,L=5),
      /**准确率:0.0708,召回率:0.0267,f1:0.0346,时间:2(ms) */
      //BasicParams(method = "Pearson",T=20,K=80,L=5),
      /**准确率:0.0630,召回率:0.0243,f1:0.0312,时间:1(ms)  */
      //T 结论：20最高
      //BasicParams(method = "Pearson",T=2,K=5,L=5),
      /**准确率:0.0404,召回率:0.0239,f1:0.0251,时间:1(ms) */
      //BasicParams(method = "Pearson",T=5,K=5,L=5),
      /**准确率:0.0666,召回率:0.0405,f1:0.0411,时间:2(ms) */
      //BasicParams(method = "Pearson",T=10,K=5,L=5),
      /**准确率:0.0856,召回率:0.0454,f1:0.0489,时间:1(ms) */
      //BasicParams(method = "Pearson",T=20,K=5,L=5),
      /**准确率:0.0858,召回率:0.0318,f1:0.0413,时间:0(ms)  */
      //BasicParams(method = "Pearson",T=40,K=5,L=5)
      /**准确率:0.0753,召回率:0.0189,f1:0.0287,时间:0(ms) */
      //Pearson总结：最优参数 T=20,K=5,L=5
      //准确率:0.0858,召回率:0.0318,f1:0.0413,时间:1(ms)
      BasicParams(method = "Pearson",T=20,K=5,L=5),

      //ImprovedPearson相似度
      // L测试 结论：5最高
      //BasicParams(method = "ImprovedPearson",T=2,K=5,L=5),
      //准确率:0.0464,召回率:0.0257,f1:0.0275,时间:2(ms)

      //BasicParams(method = "ImprovedPearson",T=2,K=5,L=10),
      //准确率:0.0441,召回率:0.0246,f1:0.0262,时间:4(ms)

      //BasicParams(method = "ImprovedPearson",T=2,K=5,L=20),
      //准确率:0.0437,召回率:0.0242,f1:0.0259,时间:2(ms)

      //BasicParams(method = "ImprovedPearson",T=2,K=5,L=40),
      //准确率:0.0434,召回率:0.0238,f1:0.0255,时间:3(ms)

      //BasicParams(method = "ImprovedPearson",T=2,K=5,L=80),
      //准确率:0.0434,召回率:0.0238,f1:0.0255,时间:3(ms)

      //K测试 结论:5最高

      //BasicParams(method = "ImprovedPearson",T=2,K=5,L=5),
      //准确率:0.0464,召回率:0.0257,f1:0.0275,时间:1(ms)

      //BasicParams(method = "ImprovedPearson",T=2,K=10,L=5),
      //准确率:0.0451,召回率:0.0245,f1:0.0264,时间:1(ms)

      //BasicParams(method = "ImprovedPearson",T=2,K=20,L=5),
      //准确率:0.0417,召回率:0.0218,f1:0.0239,时间:2(ms)

      //BasicParams(method = "ImprovedPearson",T=2,K=40,L=5),
      //准确率:0.0376,召回率:0.0189,f1:0.0211,时间:3(ms)

      //BasicParams(method = "ImprovedPearson",T=2,K=80,L=5),
      //准确率:0.0331,召回率:0.0164,f1:0.0183,时间:5(ms)

      //T 结论：10最佳

      //BasicParams(method = "ImprovedPearson",T=2,K=5,L=5),
      //准确率:0.0464,召回率:0.0257,f1:0.0275,时间:2(ms)

      //BasicParams(method = "ImprovedPearson",T=5,K=5,L=5),
      //准确率:0.0721,召回率:0.0424,f1:0.0434,时间:3(ms)

      //BasicParams(method = "ImprovedPearson",T=10,K=5,L=5),
      //准确率:0.0876,召回率:0.0461,f1:0.0493,时间:3(ms)

      //BasicParams(method = "ImprovedPearson",T=20,K=5,L=5),
      //准确率:0.0874,召回率:0.0323,f1:0.0418,时间:3(ms)

      //BasicParams(method = "ImprovedPearson",T=40,K=5,L=5)
      //准确率:0.0744,召回率:0.0187,f1:0.0284,时间:2(ms)

      //最优参数 T=10,K=5,L=5
      //准确率:0.0876,召回率:0.0461,f1:0.0493,时间:3(ms)
      BasicParams(method = "ImprovedPearson",T=10,K=5,L=5)
    )
    //结论：Jaccard的准确率最高
    //Jaccard总结：最优参数 T=2,K=5,L=5
    //准确率:0.1514,召回率:0.1060,f1:0.1029,时间:1(ms)

    for (arg <- args) {
      val recommender = new BasicRecommender(arg)
      val eval = new Evaluation()
      eval.run(recommender)
    }
  }

  def runKerasCluster():Unit={
    val args = List(
      //最优参数：K:4,maxIterations:10,K:240,L=240
      KerasClusterParams(maxIterations = 10, numNearestUsers = 240, numUserLikeMovies = 240)
    )
    for (elem <- args) {
      val recommender = new KerasClusterRecommender(elem)
      val eval = new Evaluation()
      eval.run(recommender)
    }
  }
  def runSARNCFCluster():Unit={
    val args = List(
      //最优参数：K:4,maxIterations:10,K:240,L=240
      SARNCFClusterParams(k=4,maxIterations = 10,numNearestUsers = 240,numUserLikeMovies = 240)
    )
    for (elem <- args) {
      val recommender = new SARNCFClusterRecommender(elem)
      val eval = new Evaluation()
      eval.run(recommender)
    }
  }

  def runSARCluster():Unit={
    val args = List(
      //最优参数：K:4,maxIterations:10,K:240,L=240
      SARClusterParams(CC=4,MI = 10,K = 240,L = 240)
    )
    for (elem <- args) {
      val recommender = new SARClusterRecommender(elem)
      val eval = new Evaluation()
      eval.run(recommender)
    }
  }



  def runCluster(): Unit = {
    //1.生成参数列表
    val args = List(

      /** *------------------BisectingKMeans------------------ ***/
      //BisectingKMeans
      // CC: 2 变化波动不太大
      //ClusterParams(CM = "BisectingKMeans", CC = 2, MI = 10),
      //准确率:0.0703,召回率:0.0363,f1:0.0395,时间:144(ms)
      //ClusterParams(CM = "BisectingKMeans", CC = 3, MI = 10),
      //准确率:0.0697,召回率:0.0362,f1:0.0393,时间:93(ms)
      //ClusterParams(CM = "BisectingKMeans", CC = 4, MI = 10),
      //准确率:0.0697,召回率:0.0365,f1:0.0394,时间:75(ms)
      //ClusterParams(CM = "BisectingKMeans", CC = 5, MI = 10),
      //准确率:0.0701,召回率:0.0366,f1:0.0395,时间:66(ms)
      //ClusterParams(CM = "BisectingKMeans", CC = 6, MI = 10),
      //准确率:0.0701,召回率:0.0367,f1:0.0396,时间:69(ms)
      //ClusterParams(CM = "BisectingKMeans", CC = 7, MI = 10),
      //准确率:0.0702,召回率:0.0367,f1:0.0396,时间:35(ms)
      //ClusterParams(CM = "BisectingKMeans", CC = 8, MI = 10),
      //准确率:0.0698,召回率:0.0367,f1:0.0394,时间:40(ms)
      //maxIterations:5 超过20后无变化
      //ClusterParams(CM = "BisectingKMeans", CC = 4, MI = 5),
      //准确率:0.0706,召回率:0.0367,f1:0.0399,时间:69(ms)
      //ClusterParams(CM = "BisectingKMeans", CC = 4, MI = 10),
      //准确率:0.0697,召回率:0.0365,f1:0.0394,时间:66(ms)
      //ClusterParams(CM = "BisectingKMeans", CC = 4, maxIterations = 20),
      //准确率:0.0698,召回率:0.0364,f1:0.0395,时间:79(ms)
      //ClusterParams(CM = "BisectingKMeans", CC = 4, maxIterations = 30),
      //准确率:0.0698,召回率:0.0364,f1:0.0395,时间:75(ms)
      //ClusterParams(CM = "BisectingKMeans", CC = 4, maxIterations = 40),
      //准确率:0.0698,召回率:0.0364,f1:0.0395,时间:72(ms)
      //K:240
      //ClusterParams(CM = "BisectingKMeans", CC = 4, maxIterations = 10,K=100),
      //准确率:0.1353,召回率:0.0771,f1:0.0813,时间:82(ms)
      //ClusterParams(CM = "BisectingKMeans", CC = 4, maxIterations = 10,K=120),
      //准确率:0.1385,召回率:0.0802,f1:0.0840,时间:88(ms)
      //ClusterParams(CM = "BisectingKMeans", CC = 4, maxIterations = 10,K=140),
      //准确率:0.1409,召回率:0.0823,f1:0.0860,时间:120(ms)
      //ClusterParams(CM = "BisectingKMeans", CC = 4, maxIterations = 10,K=160),
      //准确率:0.1413,召回率:0.0827,f1:0.0864,时间:51(ms)
      //ClusterParams(CM = "BisectingKMeans", CC = 4, maxIterations = 10,K=180),
      //准确率:0.1429,召回率:0.0837,f1:0.0875,时间:76(ms)
      //ClusterParams(CM = "BisectingKMeans", CC = 4, maxIterations = 10,K=200),
      //准确率:0.1445,召回率:0.0856,f1:0.0891,时间:64(ms)
      //ClusterParams(CM = "BisectingKMeans", CC = 4, maxIterations = 10,K=220),
      //准确率:0.1451,召回率:0.0868,f1:0.0899,时间:67(ms)
      //ClusterParams(CM = "BisectingKMeans", CC = 4, maxIterations = 10,K=240),
      //准确率:0.1460,召回率:0.0877,f1:0.0907,时间:68(ms)
      //ClusterParams(CM = "BisectingKMeans", CC = 4, maxIterations = 10,K=260)
      //准确率:0.1452,召回率:0.0878,f1:0.0906,时间:63(ms)
      //L:240
      //ClusterParams(CM = "BisectingKMeans", CC = 4, maxIterations = 10,K=240,L=5),
      //准确率:0.1460,召回率:0.0877,f1:0.0907,时间:70(ms)
      //ClusterParams(CM = "BisectingKMeans", CC = 4, maxIterations = 10,K=240,L=10),
      //准确率:0.1545,召回率:0.0944,f1:0.0967,时间:67(ms)
      //ClusterParams(CM = "BisectingKMeans", CC = 4, maxIterations = 10,K=240,L=20),
      //准确率:0.1622,召回率:0.0978,f1:0.1008,时间:76(ms)
      //ClusterParams(CM = "BisectingKMeans", CC = 4, maxIterations = 10,K=240,L=40),
      //准确率:0.1684,召回率:0.0984,f1:0.1027,时间:71(ms)
      //ClusterParams(CM = "BisectingKMeans", CC = 4, maxIterations = 10,K=240,L=80),
      //准确率:0.1771,召回率:0.0991,f1:0.1058,时间:74(ms)
      //ClusterParams(CM = "BisectingKMeans", CC = 4, maxIterations = 10,K=240,L=120),
      //准确率:0.1809,召回率:0.0985,f1:0.1066,时间:94(ms)
      //ClusterParams(CM = "BisectingKMeans", CC = 4, maxIterations = 10,K=240,L=160),
      //准确率:0.1825,召回率:0.0997,f1:0.1077,时间:84(ms)
      //ClusterParams(CM = "BisectingKMeans", CC = 4, maxIterations = 10,K=240,L=200),
      //准确率:0.1829,召回率:0.1001,f1:0.1081,时间:86(ms)
      //ClusterParams(CM = "BisectingKMeans", CC = 4, maxIterations = 10,K=240,L=240)//,
      //准确率:0.1833,召回率:0.1002,f1:0.1084,时间:94(ms)
      //ClusterParams(CM = "BisectingKMeans", CC = 4, maxIterations = 10,K=240,L=280)
      /**准确率:0.2175,召回率:0.1301,f1:0.1327,时间:459(ms) */

      //测试方法
      //ClusterParams(CM = "BisectingKMeans",SM="cosine", CC = 4, MI = 10,K=240,L=240),
      /**准确率:0.2176,召回率:0.1304,f1:0.1327,时间:86(ms) */
      //ClusterParams(CM = "BisectingKMeans",SM="ImprovedPearson", CC = 4, MI = 10,K=240,L=240),
      /**准确率:0.1898,召回率:0.1275,f1:0.1249,时间:72(ms) */
        //ClusterParams(CM = "BisectingKMeans",SM="pearson", CC = 4, MI = 10,K=240,L=240)//,
      /**准确率:0.2169,召回率:0.1305,f1:0.1327,时间:73(ms) */

      /** 最优参数：CC:4,MI:10,K:240,L=240
       * 最好结果：准确率:0.2175,召回率:0.1301,f1:0.1327,时间:459(ms)*/
      ClusterParams(CM = "BisectingKMeans",SM="cosine", CC = 4, MI = 10,K=240,L=240)//,

      /** *------------------K-means------------------ ***/
      //maxIterations:10
      //ClusterParams(CM = "K-means", k = 4, maxIterations = 5, K = 240, L = 240)//,
      /**准确率:0.2165,召回率:0.1285,f1:0.1317,时间:92(ms) */

      //ClusterParams(CM = "K-means", k = 4, maxIterations = 10, K = 240, L = 240),
      //准确率:0.1849,召回率:0.1014,f1:0.1092,时间:112(ms)

      //ClusterParams(CM = "K-means", k = 4, maxIterations = 20, K = 240, L = 240),
      //准确率:0.1845,召回率:0.1000,f1:0.1085,时间:121(ms)

      //ClusterParams(CM = "K-means", k = 4, maxIterations = 40, K = 240, L = 240),
      //准确率:0.1844,召回率:0.1000,f1:0.1085,时间:110(ms)

      //ClusterParams(CM = "K-means", k = 4, maxIterations = 80, K = 240, L = 240)
      //准确率:0.1846,召回率:0.1008,f1:0.1088,时间:114(ms)

      //k:4
      //ClusterParams(CM = "K-means", k = 3, maxIterations = 10, K = 240, L = 240),
      //准确率:0.1840,召回率:0.0996,f1:0.1082,时间:94(ms)

      //ClusterParams(CM = "K-means", k = 4, maxIterations = 10, K = 240, L = 240),
      //准确率:0.1846,召回率:0.1007,f1:0.1087,时间:79(ms)

      //ClusterParams(CM = "K-means", k = 5, maxIterations = 10, K = 240, L = 240),
      //准确率:0.1818,召回率:0.1019,f1:0.1086,时间:101(ms)

      //ClusterParams(CM = "K-means", k = 6, maxIterations = 10, K = 240, L = 240),
      //准确率:0.1842,召回率:0.1044,f1:0.1109,时间:69(ms)

      //ClusterParams(CM = "K-means", k = 7, maxIterations = 10, K = 240, L = 240)
      //准确率:0.1824,召回率:0.1028,f1:0.1094,时间:71(ms)

      //K-means最优参数：k = 4, maxIterations = 10, K = 240, L = 240
      //准确率:0.1849,召回率:0.1014,f1:0.1092,时间:112(ms)


      //Cosine,ImprovedPearson,Pearson
      //ClusterParams(CM = "K-means",SM="Cosine", k = 7, maxIterations = 10, K = 240, L = 240),
      /**准确率:0.2151,召回率:0.1301,f1:0.1324,时间:75(ms) */
      //ClusterParams(CM = "K-means",SM="ImprovedPearson", k = 7, maxIterations = 10, K = 240, L = 240),
      /**准确率:0.1889,召回率:0.1274,f1:0.1246,时间:61(ms) */

      //ClusterParams(CM = "K-means",SM="Pearson", k = 7, maxIterations = 10, K = 240, L = 240)
      /**准确率:0.2146,召回率:0.1296,f1:0.1321,时间:57(ms) */


      /** *------------------GaussianMixture------------------ ***/
      //GaussianMixture
      //k:2
      //ClusterParams(CM = "GaussianMixture", k = 2, maxIterations = 10, K = 240, L = 240),
      //准确率:0.1861,召回率:0.0972,f1:0.1071,时间:269(ms)
      //ClusterParams(CM = "GaussianMixture", k = 3, maxIterations = 10, K = 240, L = 240),
      //准确率:0.1861,召回率:0.0971,f1:0.1070,时间:302(ms)
      //ClusterParams(CM = "GaussianMixture", k = 4, maxIterations = 10, K = 240, L = 240),
      /**准确率:0.2140,召回率:0.1275,f1:0.1305,时间:92(ms) */
      //ClusterParams(CM = "GaussianMixture", k = 5, maxIterations = 10, K = 240, L = 240),
      //准确率:0.1858,召回率:0.0966,f1:0.1066,时间:187(ms)
      //ClusterParams(CM = "GaussianMixture", k = 6, maxIterations = 10, K = 240, L = 240),
      //准确率:0.1853,召回率:0.0965,f1:0.1065,时间:153(ms)
      //ClusterParams(CM = "GaussianMixture", k = 7, maxIterations = 10, K = 240, L = 240),
      //准确率:0.1853,召回率:0.0963,f1:0.1065,时间:155(ms)
      //maxIterations:10
      //ClusterParams(CM = "GaussianMixture", k = 2, maxIterations = 10, K = 240, L = 240),
      //准确率:0.1862,召回率:0.0973,f1:0.1071,时间:220(ms)
      //ClusterParams(CM = "GaussianMixture", k = 2, maxIterations = 20, K = 240, L = 240),
      //准确率:0.1861,召回率:0.0972,f1:0.1071,时间:210(ms)
      //ClusterParams(CM = "GaussianMixture", k = 2, maxIterations = 30, K = 240, L = 240),
      //准确率:0.1862,召回率:0.0971,f1:0.1071,时间:177(ms)
      //ClusterParams(CM = "GaussianMixture", k = 2, maxIterations = 40, K = 240, L = 240),
      //准确率:0.1861,召回率:0.0971,f1:0.1070,时间:188(ms)
      //ClusterParams(CM = "GaussianMixture", k = 2, maxIterations = 50, K = 240, L = 240)
      //准确率:0.1861,召回率:0.0973,f1:0.1071,时间:206(ms)

      //GaussianMixture:簇心数量:2,maxIterations:10,相似度方法:Cosine,K:240,L:240
      //准确率:0.1862,召回率:0.0973,f1:0.1071,时间:220(ms)

      //ImprovedPearson
      //ClusterParams(CM = "GaussianMixture",SM="Cosine", k = 2, maxIterations = 50, K = 240, L = 240),
      /**准确率:0.2140,召回率:0.1224,f1:0.1275,时间:156(ms) */
      //ClusterParams(CM = "GaussianMixture", SM="ImprovedPearson",k = 2, maxIterations = 50, K = 240, L = 240),
      /**准确率:0.1906,召回率:0.1278,f1:0.1253,时间:91(ms) */
      //ClusterParams(CM = "GaussianMixture",SM="Pearson", k = 2, maxIterations = 50, K = 240, L = 240)
      /**准确率:0.2143,召回率:0.1242,f1:0.1285,时间:89(ms) */

    )
    for (arg <- args) {
      val recommender = new ClusterRecommender(arg)
      val eval = new Evaluation()
      eval.run(recommender)
    }
  }

  def runNCFCluster(): Unit = {

    val args = List(
      //NCFClusterParams(maxIterations = 10, K = 240, L = 240),
      NCFClusterParams(oneHot=true, maxIterations = 10, numNearestUsers = 240, numUserLikeMovies = 240)
      //NCFClusterParams(method = "ImprovedPearson", maxIterations = 10, K = 240, L = 240),
      //NCFClusterParams(method = "pearson", maxIterations = 10, K = 240, L = 240)
    )
    for (elem <- args) {
      val recommender = new NCFClusterRecommender(elem)
      val eval = new Evaluation()
      eval.run(recommender)
    }

  }

  def runRandomClusterForest(): Unit = {


    val args = List(
      /** BisectingKMeans K:4,maxIterations:10,K:240,L=240
        * 计算相似度方法:cosine:T=5,K=5,L=5
        */
      RandomForestClusterParams(k = 4,numNearestUsers = 240,numUserLikeMovies=240,maxIterationsCluster=10)//,
      //准确率:0.2176,召回率:0.1304,f1:0.1327,时间:102(ms)


      /**
        * RandomForestClusterParams:聚类部分：{邻近用户数量：240,L:240,
        * 计算相似度方法：improvedpearson,聚类中心数量:4}
        */
      //RandomForestClusterParams(method = "improvedpearson",k = 4,K = 240,L=240,maxIterationsCluster=10),
      //准确率:0.1902,召回率:0.1276,f1:0.1250,时间:152(ms)
      /**
        * RandomForestClusterParams:聚类部分：{邻近用户数量：240,L:240,计算相似度方法：pearson,
        * 聚类中心数量:4}
        * 随机森林部分：{最大迭代次数:20,分类数量:2,子数数量:5,子树分割策略:auto,impurity:gini,最大数深:5,maxBins:100}
        **/
      //RandomForestClusterParams(method = "pearson",k = 4,K = 240,L=240,maxIterationsCluster=10)
      //准确率:0.2170,召回率:0.1305,f1:0.1327,时间:113(ms)

      //RandomForestClusterParams(K = 240, L = 240, maxIterationsCluster = 10, numTrees = 5),
      /**准确率:0.2176,召回率:0.1304,f1:0.1327,时间:147(ms) */
      //RandomForestClusterParams(K = 240, L = 240, maxIterationsCluster = 10, numTrees = 7),
      /**准确率:0.2176,召回率:0.1304,f1:0.1327,时间:164(ms) */
      //RandomForestClusterParams(K = 240, L = 240, maxIterationsCluster = 10, numTrees = 9)
      /**准确率:0.2176,召回率:0.1303,f1:0.1327,时间:152(ms)  */
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




}
