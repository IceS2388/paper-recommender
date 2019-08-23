package recommender.tools

import scala.collection.immutable.HashMap

/**
  * Author:IceS
  * Date:2019-08-21 13:53:46
  * Description:
  * NONE
  */
class BiMap[K, V](
                   private val m: Map[K, V],
                   private val i: Option[BiMap[V, K]] = None
                 ) extends Serializable {

  // NOTE: make inverse's inverse point back to current recommender.tools.BiMap
  val inverse: BiMap[V, K] = i.getOrElse {
    val rev = m.map(_.swap)
    require(rev.size == m.size,
      s"Failed to create reversed map. Cannot have duplicated values.")
    new BiMap(rev, Some(this))
  }

  def get(k: K): Option[V] = m.get(k)

  def getOrElse(k: K, default: => V): V = m.getOrElse(k, default)

  def contains(k: K): Boolean = m.contains(k)

  def apply(k: K): V = m.apply(k)

  /** Converts to a map.
    *
    * @return a map of type immutable.Map[K, V]
    */
  def toMap: Map[K, V] = m

  /** Converts to a sequence.
    *
    * @return a sequence containing all elements of this map
    */
  def toSeq: Seq[(K, V)] = m.toSeq

  def size: Int = m.size

  def take(n: Int): BiMap[K, V] = BiMap(m.take(n))

  override def toString: String = m.toString
}

object BiMap {

  def apply[K, V](x: Map[K, V]): BiMap[K, V] = new BiMap(x)

  /** Create a recommender.tools.BiMap[Int, Long] from a set of Int. The Int index starts
    * from 0.
    *
    * @param keys a set of Int
    * @return a Int to Long recommender.tools.BiMap
    */
  def toIndex(keys: Set[Int]): BiMap[Int, Long] = {
    val hm = HashMap(keys.toSeq.zipWithIndex.map(t => (t._1, t._2.toLong)): _*)
    new BiMap(hm)
  }

  def toIndex(keys: Seq[Int]): BiMap[Int, Long] = {
    val hm = HashMap(keys.zipWithIndex.map(t => (t._1, t._2.toLong)): _*)
    new BiMap(hm)
  }

}
