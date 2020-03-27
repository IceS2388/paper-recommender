package recommender

abstract class Recommender {
  def getParams: Params
  def prepare(data: Seq[Rating]): PreparedData
  def train(data: TrainingData): Unit
  def predict(query: Query): PredictedResult
}
