package dr.acf.experiments

import java.io.{File, IOException}
import java.text.DecimalFormat
import java.util

import dr.acf.utils.{SmartEvaluation, SparkOps}
import org.apache.spark.rdd.RDD
import org.datavec.api.records.reader.RecordReader
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.writable.Writable
import org.datavec.spark.transform.misc.StringToWritablesFunction
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf._
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.spark.api.RDDTrainingApproach
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.deeplearning4j.ui.storage.FileStatsStorage
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j
import org.slf4j.LoggerFactory

import scala.collection.JavaConversions._
import scala.util.Try

/**
  * Created by acflorea on 15/10/2016.
  */
object BagOfWords extends SparkOps {

  val log: slf4j.Logger = LoggerFactory.getLogger(Classifier.getClass)

  def main(args: Array[String]): Unit = {

    // Location of resource files.
    val resourceFolder = conf.getString("global.resourceFolder")
    // Input File names
    val trainFileName = conf.getString("global.trainFileName")
    val testFileName = conf.getString("global.testFileName")
    // Type of the architecture to use
    val architecture = conf.getString("global.architecture")
    // Initial epoch value
    val startEpoch = conf.getInt("global.startEpoch")
    // Source model to start from
    val sourceModel = conf.getString("global.sourceModel")

    //Execute training:
    val numEpochs = if (conf.hasPath("global.numEpochs")) conf.getInt("global.numEpochs") else 100

    // Batch Size
    val batchSize = if (conf.hasPath("global.batchSize")) conf.getInt("global.batchSize") else 25
    // Averaging Frequency
    val averagingFrequency = if (conf.hasPath("global.averagingFrequency")) conf.getInt("global.averagingFrequency") else 5

    log.debug(s"Spark Master is ${sc.master}")
    log.debug(s"Default parallelism is ${sc.defaultParallelism}")

    log.debug(s"Resource folder is $resourceFolder")
    log.debug(s"Train file is $testFileName")
    log.debug(s"Test file is $testFileName")

    log.debug(s"Batch Size is $batchSize")
    log.debug(s"Averaging Frequency is $averagingFrequency")

    val height = architecture match {
      case "cnn" => 5
      case "rnn" => 1
      case "deep" => 1
    }


    val trainDirectory: String = resourceFolder + trainFileName
    val testDirectory: String = resourceFolder + testFileName

    val trainStringData: RDD[String] = sc.textFile(s"file:$trainDirectory")
    val testStringData: RDD[String] = sc.textFile(s"file:$testDirectory")
    val trainBatchSize = trainStringData.count().toInt / sc.defaultParallelism / 15

    //We first need to parse this format. It's comma-delimited (CSV) format, so let's parse it using CSVRecordReader:
    val train_rr: RecordReader = new CSVRecordReader(0, CSVRecordReader.DEFAULT_DELIMITER)
    val trainParsedInputData: RDD[util.List[Writable]] = trainStringData.map(new StringToWritablesFunction(train_rr).call(_))
    val possibleLabels = trainParsedInputData.map(_.last).collect().toList.distinct.size

    val seed = 12345

    val featureSpaceSize = trainParsedInputData.take(1).last.size - 1

    val outputNum = possibleLabels

    val iterations = 500

    val layer1width = 250
    val learningRate = 0.0018
    val activation = "softsign"

    val activation_end = "softmax"

    val net = if (sourceModel.trim == "") {

      log.info("Build model....")
      log.info(s"Number of iterations $iterations")
      log.info(s"Number of features $featureSpaceSize")
      log.info(s"Number of labels $possibleLabels")

      val cnn_conf: Option[MultiLayerConfiguration] = Try(new NeuralNetConfiguration.Builder()
        .seed(seed)
        .seed(12345).iterations(iterations).regularization(true).l2(0.0005).learningRate(.01)
        .weightInit(WeightInit.XAVIER).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .updater(Updater.NESTEROVS).momentum(0.9)
        .list
        .layer(0, new ConvolutionLayer.Builder(2, 1).name("conv1")
          // nIn is the number of channels, nOut is the number of filters to be applied
          .nIn(1).stride(1, 1).nOut(20)
          .activation("identity").build())
        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG).name("pooling_1")
          .kernelSize(2, 1).stride(1, 1).build())
        .layer(2, new ConvolutionLayer.Builder(2, 1).name("conv2")
          .stride(1, 1).nOut(20)
          .activation("identity").build())
        .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG).name("pooling_2")
          .kernelSize(2, 1).stride(1, 1).build())
        .layer(4, new DenseLayer.Builder().activation("relu").nOut(500).build())
        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(outputNum).activation("softmax").build())
        // height, width, height
        .setInputType(InputType.convolutionalFlat(height, featureSpaceSize / height, 1))
        .backprop(true).pretrain(false).build()).toOption

      val cnn_conf_2: Option[MultiLayerConfiguration] = Try(new NeuralNetConfiguration.Builder()
        .seed(seed)
        .seed(12345).iterations(iterations).regularization(true).l2(0.0005).learningRate(.01)
        .weightInit(WeightInit.XAVIER).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .updater(Updater.NESTEROVS).momentum(0.9)
        .list
        .layer(0, new ConvolutionLayer.Builder(height, 1).name("conv1")
          // nIn is the number of channels, nOut is the number of filters to be applied
          .nIn(1).stride(1, 1).nOut(20)
          .activation("identity").build())
        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG).name("pooling_1")
          .kernelSize(1, 1).stride(1, 1).build())
        .layer(2, new ConvolutionLayer.Builder(1, 1).name("conv2")
          .dropOut(0.5)
          .stride(1, 1).nOut(50).activation("identity").build())
        .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG).name("pooling_2")
          .kernelSize(1, 1).stride(1, 1).build())
        .layer(4, new DenseLayer.Builder().activation("relu").nOut(500).build())
        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(outputNum).activation("softmax").build())
        // height, width, height
        .setInputType(InputType.convolutionalFlat(height, featureSpaceSize, 1))
        .backprop(true).pretrain(false).build()).toOption

      //Set up network configuration
      val rnn_conf: Option[MultiLayerConfiguration] = Try(new NeuralNetConfiguration.Builder()
        .seed(seed)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .iterations(iterations)
        .updater(Updater.RMSPROP)
        .regularization(true).l2(1e-5)
        .weightInit(WeightInit.XAVIER)
        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
        .learningRate(learningRate)
        .list
        .layer(0, new GravesLSTM.Builder().name("GravesLSTM")
          .nIn(featureSpaceSize).nOut(layer1width)
          .activation(activation)
          .dropOut(0.5)
          .build())
        .layer(1, new RnnOutputLayer.Builder().name("RnnOutputLayer")
          .activation(activation_end)
          .nIn(layer1width).nOut(outputNum)
          .build())
        .pretrain(false).backprop(true).build).toOption

      val deep_conf: Option[MultiLayerConfiguration] = Try(new NeuralNetConfiguration.Builder()
        .seed(seed)
        .iterations(iterations)
        .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
        .list
        .layer(0, new RBM.Builder().nIn(featureSpaceSize).nOut(500).lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).build)
        //.layer(1, new RBM.Builder().nIn(10).nOut(10).lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).build)
        //.layer(2, new RBM.Builder().nIn(10).nOut(10).lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).build)
        //.layer(3, new RBM.Builder().nIn(10).nOut(10).lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).build)
        //.layer(4, new RBM.Builder().nIn(10).nOut(10).lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).build)
        .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation("softmax").nIn(500).nOut(outputNum).build)
        .pretrain(true).backprop(true).build).toOption

      log.info(s"Architecture $architecture")
      val active_conf = architecture match {
        case "cnn" => cnn_conf.get
        case "rnn" => rnn_conf.get
        case "deep" => deep_conf.get
      }

      log.info(s"Network configuration ${active_conf.toString}")

      def _net = new MultiLayerNetwork(active_conf)

      _net.init()

      _net

    } else {

      //Load the model
      val locationToSave = new File(s"$sourceModel")
      val restored = ModelSerializer.restoreMultiLayerNetwork(locationToSave)
      restored.init()

      restored

    }

    val _trainData = CSVDatasetIterator(trainDirectory, trainBatchSize, featureSpaceSize, possibleLabels).toList
    val _testData = CSVDatasetIterator(testDirectory, testStringData.count().toInt, featureSpaceSize, possibleLabels).toList

    log.info(s"Training data size ${_trainData.length}")
    log.info(s"Test data size ${_testData.length}")

    val saveUpdater = true

    val trainingData = sc.parallelize(_trainData)
    val testData = sc.parallelize(_testData)

    val tm = new ParameterAveragingTrainingMaster.Builder(trainBatchSize)
      .rddTrainingApproach(RDDTrainingApproach.Export)
      .averagingFrequency(averagingFrequency)
      .workerPrefetchNumBatches(2)
      .batchSizePerWorker(batchSize)
      .build()

    val networkListeners = new util.ArrayList[IterationListener]()
    networkListeners.add(new ScoreIterationListener(iterations / 5))

    val sparkNet = new SparkDl4jMultiLayer(sc, net, tm)
    sparkNet.setCollectTrainingStats(true)

    def statsStorage = new FileStatsStorage(new File("trainingStats.dl4j"))

    sparkNet.setListeners(statsStorage, networkListeners)

    log.info(s"Start training!!!")

    log.info("Epoch," +
      "Accuracy_Train,Precision_Train,Recall_Train,F1Score_Train,WPrecision_Train,WRecall_Train,WF1Score_Train," +
      "Accuracy_Test,Precision_Test,Recall_Test,F1Score_Test,WPrecision_Test,WRecall_Test,WF1Score_Test")

    val df: DecimalFormat = new DecimalFormat("0000")

    (startEpoch to numEpochs) foreach { i =>
      sparkNet.fit(trainingData)

      val locationToSave = new File(s"${architecture}_${trainFileName}_$i.zip")
      ModelSerializer.writeModel(net, locationToSave, saveUpdater)

      val evaluationTrain: SmartEvaluation = new SmartEvaluation(sparkNet.evaluate(trainingData))
      val evaluationTest: SmartEvaluation = new SmartEvaluation(sparkNet.evaluate(testData))

      log.info(s"${df.format(i)},${evaluationTrain.csvStats},${evaluationTest.csvStats}")
    }

    //Delete the temp training files, now that we are done with them
    tm.deleteTempFiles(sc)

  }

  /**
    * used for testing and training
    *
    * @param csvFileLocation
    * @param batchSize
    * @param labelIndex
    * @param numClasses
    * @return
    * @throws IOException
    * @throws InterruptedException
    */
  @throws[IOException]
  @throws[InterruptedException]
  private def CSVDatasetIterator(csvFileLocation: String, batchSize: Int, labelIndex: Int, numClasses: Int) = {
    val rr = new CSVRecordReader
    rr.initialize(new FileSplit(new File(csvFileLocation)))
    val iterator = new RecordReaderDataSetIterator(rr, batchSize, labelIndex, numClasses)
    iterator
  }

}
