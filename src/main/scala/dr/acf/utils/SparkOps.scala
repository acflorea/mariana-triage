package dr.acf.utils

import org.apache.log4j.{Level, Logger}
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
  conf.set("spark.driver.maxResultSize", "2g")

  lazy val sc: SparkContext = {
    val _sc = new SparkContext(conf)

    val rootLogger = Logger.getRootLogger
    rootLogger.setLevel(Level.WARN)

    _sc
  }

}
