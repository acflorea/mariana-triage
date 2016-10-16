package dr.acf.utils

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Spark Context related functionality
  * Created by acflorea on 16/10/2016.
  */
trait SparkOps {

  //We'll use Spark local to handle our data
  private val conf: SparkConf = new SparkConf
  conf.setMaster("local[*]")
  conf.setAppName("mariana-triage")

  lazy val sc: SparkContext = new SparkContext(conf)

}