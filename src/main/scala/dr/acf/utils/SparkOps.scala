package dr.acf.utils

import com.typesafe.config.ConfigFactory
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Spark Context related functionality
  * Created by acflorea on 16/10/2016.
  */
trait SparkOps {

  val conf = ConfigFactory.load().getConfig("mariana")

  lazy val sc: SparkContext = {
    val master = conf.getString("spark.master")
    val appName = conf.getString("spark.appName")

    val sparkConf = new SparkConf().setAppName(appName)
    sparkConf.set("spark.driver.maxResultSize", conf.getString("spark.driver.maxResultSize"))

    // JavaCPP
    sparkConf.set("spark.executor.extraJavaOptions", "-Dorg.bytedeco.javacpp.maxbytes=5368709120")
    sparkConf.set("spark.driver.extraJavaOptions", "-Dorg.bytedeco.javacpp.maxbytes=5368709120")

    new SparkContext(sparkConf)

  }

}
