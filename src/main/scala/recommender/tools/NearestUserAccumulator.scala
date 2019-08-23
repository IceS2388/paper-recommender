package recommender.tools

import scala.collection.mutable

/**
  * Author:IceS
  * Date:2019-08-20 19:20:09
  * Description:
  * NONE
  */
class NearestUserAccumulator {
  private val mapAccumulator = mutable.Map[String, Double]()

  def containsKey(k1: Int, k2: Int): Boolean = {
    val key1 = s",$k1,$k2,"
    val key2 = s",$k2,$k1,"
    mapAccumulator.contains(key1) || mapAccumulator.contains(key2)
  }


  def add(v: (Int, Int, Double)): Unit = {
    val u1 = v._1
    val u2 = v._2
    val score = v._3
    if (!this.containsKey(u1, u2)) {
      val key = s",$u1,$u2,"
      mapAccumulator += key -> score
    } else {
      val key1 = s",$u1,$u2,"
      val key2 = s",$u2,$u1,"
      if (mapAccumulator.contains(key1)) {
        mapAccumulator.put(key1, score)
      } else if (mapAccumulator.contains(key2)) {
        mapAccumulator.put(key2, score)
      }
    }

  }

  def add(v: (String, Double)): Unit = {
    val key = v._1
    val value = v._2
    if (!mapAccumulator.contains(key))
      mapAccumulator += key -> value
    else
      mapAccumulator.put(key, value)
  }

  def value: mutable.Map[String, Double] = {
    mapAccumulator
  }
}
