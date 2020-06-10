package recommender
/**
  * 抽象类，用于定义所有具体推荐算法必须实现的方法。
  * */
abstract class Recommender {
  def getParams: Params
  def prepare(data: Seq[Rating]): PreparedData
  def train(data: TrainingData): Unit
  def predict(query: Query): PredictedResult
}
