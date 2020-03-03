package recommender

/**
  * 原始数据分割后，用于训练的数据。
  **/
class TrainingData(val ratings: Seq[Rating]) {
  override def toString = {
    s"TrainingData: [${ratings.size}] (${ratings.take(2).toList}...)"
  }
}
