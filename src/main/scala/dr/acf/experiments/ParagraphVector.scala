package dr.acf.experiments

import java.io.File
import java.text.DecimalFormat
import java.util
import java.util.{Collections, TimeZone}

import dr.acf.utils.{SmartEvaluation, SparkOps, WordVectorSmartSerializer}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.datavec.api.records.reader.RecordReader
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.transform.TransformProcess
import org.datavec.api.transform.schema.Schema
import org.datavec.api.writable.{DoubleWritable, Writable}
import org.datavec.spark.transform.SparkTransformExecutor
import org.datavec.spark.transform.misc.StringToWritablesFunction
import org.deeplearning4j.api.storage.StatsStorage
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors
import org.deeplearning4j.models.sequencevectors.SequenceVectors
import org.deeplearning4j.models.sequencevectors.enums.ListenerEvent
import org.deeplearning4j.models.sequencevectors.interfaces.VectorsListener
import org.deeplearning4j.models.word2vec.VocabWord
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
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.FileStatsStorage
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j
import org.slf4j.LoggerFactory

import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.util.Try

/**
  * Created by acflorea on 15/10/2016.
  */
object ParagraphVector extends SparkOps {

  val severityValues: util.List[String] = util.Arrays.asList("normal", "enhancement", "major", "trivial", "critical", "minor", "blocker")
  val statusValues: util.List[String] = util.Arrays.asList("CLOSED", "RESOLVED", "VERIFIED", "trivial", "critical", "minor")

  // Let's define the schema of the data that we want to import
  // The order in which columns are defined here should match
  // the order in which they appear in the input data
  val inputDataSchema: Schema = new Schema.Builder()
    .addColumnInteger("row_id")
    .addColumnInteger("assigned_to")
    .addColumnInteger("bug_id")
    .addColumnCategorical("bug_severity", severityValues)
    .addColumnCategorical("bug_status", statusValues)
    .addColumnInteger("component_id")
    .addColumnTime("creation_ts", TimeZone.getDefault)
    .addColumnTime("delta_ts", TimeZone.getDefault)
    .addColumnInteger("product_id")
    .addColumnCategorical("resolution", util.Arrays.asList("FIXED"))
    .addColumnsString("short_desc", "original_text", "text")
    .addColumnInteger("class")
    .build()

  val filteredDataSchema: Schema = new Schema.Builder()
    .addColumnsString("text")
    .addColumnCategorical("bug_severity", severityValues)
    .addColumnInteger("component_id")
    .addColumnInteger("product_id")
    .addColumnInteger("class")
    .build()

  val log: slf4j.Logger = LoggerFactory.getLogger(Classifier.getClass)

  def main(args: Array[String]): Unit = {

    // Location of resource files.
    val resourceFolder = conf.getString("global.resourceFolder")
    // Input File name
    val inputFileName = conf.getString("global.inputFileName")
    // Weather to compute the embeddings or attempt to load an existing model
    val computeEmbeddings = conf.getBoolean("global.computeEmbeddings")
    // Number of epochs for embeddings
    val epochsForEmbeddings = conf.getInt("global.epochsForEmbeddings")
    // Name of model file to user
    val model = conf.getString("global.model")
    // Type of the architecture to use
    val architecture = conf.getString("global.architecture")
    // Source model to start from
    val sourceModel = conf.getString("global.sourceModel")
    // Initial epoch value
    val startEpoch = conf.getInt("global.startEpoch")

    // Batch Size
    val batchSize = if (conf.hasPath("global.batchSize")) conf.getInt("global.batchSize") else 25
    // Averaging Frequency
    val averagingFrequency = if (conf.hasPath("global.averagingFrequency")) conf.getInt("global.averagingFrequency") else 5

    lazy val modelName = if (epochsForEmbeddings < 10)
      s"${model}_0$epochsForEmbeddings.model"
    else
      s"${model}_$epochsForEmbeddings.model"

    log.debug(s"Spark Master is ${sc.master}")
    log.debug(s"Default parallelism is ${sc.defaultParallelism}")

    log.debug(s"Resource folder is $resourceFolder")
    log.debug(s"Input file is $inputFileName")
    log.debug(s"Compute embeddings is $computeEmbeddings")
    log.debug(s"Number of embedding epochs is $epochsForEmbeddings")

    log.debug(s"Batch Size is $batchSize")
    log.debug(s"Averaging Frequency is $averagingFrequency")

    //Print out the schema:
    log.info("Input data schema details:")
    log.info(s"$inputDataSchema")

    log.info("\n\nOther information obtainable from schema:")
    log.info(s"Number of columns: ${inputDataSchema.numColumns}")
    log.info(s"Column names: ${inputDataSchema.getColumnNames}")
    log.info(s"Column types: ${inputDataSchema.getColumnTypes}")

    //=====================================================================
    //            Step 2.a: Define the operations we want to do
    //=====================================================================
    val filterColumnsTransform: TransformProcess = new TransformProcess.Builder(inputDataSchema)
      //Let's remove some column we don't need
      //  .filter(new ConditionFilter(new IntegerColumnCondition("class", ConditionOp.GreaterOrEqual, 20)))
      //  .filter(new ConditionFilter(new IntegerColumnCondition("component_id", ConditionOp.NotEqual, 128)))
      .removeAllColumnsExceptFor("text", "bug_severity", "component_id", "product_id", "class")
      .reorderColumns("text", "bug_severity", "component_id", "product_id", "class")
      //.removeAllColumnsExceptFor("text", "class")
      //.reorderColumns("text", "class")
      .build()

    // After executing all of these operations, we have a new and different schema:
    val outputSchema: Schema = filterColumnsTransform.getFinalSchema

    log.info("\n\n\nSchema after transforming data:")
    log.info(s"$outputSchema")

    //=====================================================================
    //            Step 2.b: Transform
    //=====================================================================
    val directory: String = resourceFolder + inputFileName
    val stringData: RDD[String] = sc.textFile(s"file:$directory")

    //We first need to parse this format. It's comma-delimited (CSV) format, so let's parse it using CSVRecordReader:
    val rr: RecordReader = new CSVRecordReader(0, CSVRecordReader.DEFAULT_DELIMITER)
    val parsedInputData: RDD[util.List[Writable]] = stringData.map(new StringToWritablesFunction(rr).call(_))

    //Now, let's execute the transforms we defined earlier:
    val filteredData: RDD[util.List[Writable]] = SparkTransformExecutor.execute(parsedInputData, filterColumnsTransform)

    val components: java.util.Map[java.lang.Integer, java.lang.String] =
      new java.util.HashMap[java.lang.Integer, java.lang.String]()

    val products: java.util.Map[java.lang.Integer, java.lang.String] =
      new java.util.HashMap[java.lang.Integer, java.lang.String]()

    val classes: java.util.Map[java.lang.Integer, java.lang.String] =
      new java.util.concurrent.ConcurrentHashMap[java.lang.Integer, java.lang.String]()
    val distinctClasses = mutable.Set.empty[Int]

    val descs = filteredData.collect().map {
      writables =>
        if (Try(writables.get(2).toInt).isSuccess) components.put(writables.get(2).toInt, writables.get(2).toString)
        if (Try(writables.get(3).toInt).isSuccess) products.put(writables.get(3).toInt, writables.get(3).toString)
        if (Try(writables.last.toInt).isSuccess) distinctClasses += writables.last.toInt
        writables.get(0).toString
    }

    distinctClasses.zipWithIndex foreach { classWithIndex =>
      classes.put(classWithIndex._1, classWithIndex._2.toString)
    }

    //=====================================================================
    //            Step 3.a: More transformations
    //=====================================================================
    val numericToCategoricalTransform: TransformProcess = new TransformProcess.Builder(filteredDataSchema)
      .integerToCategorical("component_id", components)
      .integerToCategorical("product_id", products)
      .categoricalToOneHot("bug_severity")
      .categoricalToOneHot("component_id")
      .categoricalToOneHot("product_id")
      .integerToCategorical("class", classes)
      .build()

    //=====================================================================
    //            Step 3.b: Transform
    //=====================================================================
    //Now, let's execute the transforms we defined earlier:
    val transformedData: RDD[util.List[Writable]] = SparkTransformExecutor.execute(filteredData, numericToCategoricalTransform)

    val possibleLabels: Int = classes.size()

    //=====================================================================
    //            PARAGRAPH VECTOR !!!
    //=====================================================================

    // build a iterator for our dataset
    val ptvIterator = new CollectionSentenceIterator(descs.flatMap(_.split("\\. ")).toList)
    val prvRDD = sc.parallelize(descs.flatMap(_.split("\\. ")).toList)

    //
    val tokenizerFactory = new DefaultTokenizerFactory
    tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor)

    val paragraphVectors = if (computeEmbeddings) {

      // Lister to store model at each phase
      val vectorListener = new VectorsListener[VocabWord] {
        override def processEvent(event: ListenerEvent, sequenceVectors: SequenceVectors[VocabWord], argument: Long): Unit = {
          event match {
            case ListenerEvent.EPOCH if argument % 10 == 0 =>
              log.info("Save vectors....")
              lazy val _modelName = if (argument < 10)
                s"${model}_0$argument.model"
              else
                s"${model}_$argument.model"
              WordVectorSmartSerializer
                .writeParagraphVectors(sequenceVectors.asInstanceOf[ParagraphVectors], resourceFolder + modelName)
            case _ =>
          }
        }

        override def validateEvent(event: ListenerEvent, argument: Long): Boolean = true
      }

      val listeners = new util.ArrayList[VectorsListener[VocabWord]]()
      listeners.add(vectorListener)

      log.info("Build Embedded Vectors ....")

      // ParagraphVectors training configuration
      val _paragraphVectors = if (sourceModel != "") {
        // start with initial vactors
        log.info("Load Initial Vectors ....")
        val _initialPV = WordVectorSmartSerializer.readParagraphVectors(new File(resourceFolder + sourceModel))
        new ParagraphVectors.Builder()
          .useExistingWordVectors(_initialPV)
          .setVectorsListeners(listeners)
          .layerSize(100)
          .learningRate(0.025).minLearningRate(0.001)
          .batchSize(2500).epochs(epochsForEmbeddings)
          .iterate(ptvIterator).trainWordVectors(true)
          .tokenizerFactory(tokenizerFactory).build
      } else {
        new ParagraphVectors.Builder()
          .setVectorsListeners(listeners)
          .layerSize(100)
          .learningRate(0.025).minLearningRate(0.001)
          .batchSize(2500).epochs(epochsForEmbeddings)
          .iterate(ptvIterator).trainWordVectors(true)
          .tokenizerFactory(tokenizerFactory).build
      }

      // Start model training
      _paragraphVectors.fit()

      log.info("Save vectors....")
      WordVectorSmartSerializer.writeParagraphVectors(_paragraphVectors, resourceFolder + modelName)
      _paragraphVectors

    } else {

      log.info("Load Embedded Vectors ....")
      val _paragraphVectors = WordVectorSmartSerializer.readParagraphVectors(new File(resourceFolder + modelName))
      _paragraphVectors.setTokenizerFactory(tokenizerFactory)
      _paragraphVectors

    }

    log.info("Embedded Vectors OK ....")

    val height = architecture match {
      case "cnn" => 5
      case "rnn" => 1
      case "deep" => 1
    }

    log.info("Apply Paragraph2Vec ....")

    val broadcastPV = true
    val data =
      if (broadcastPV) {
        val pv = sc.broadcast(paragraphVectors)
        transformedData.repartition(sc.defaultParallelism).mapPartitions { it =>
          it.zipWithIndex.collect { case rowWithIndex if rowWithIndex._1.head.toString.nonEmpty =>
            addWord2VecPartitions(pv, height, rowWithIndex)
          }
        }.collect().toList
      } else {
        transformedData.collect().zipWithIndex.toList.collect { case rowWithIndex if rowWithIndex._1.head.toString.nonEmpty =>
          addWord2Vec(paragraphVectors, height, rowWithIndex)
        }
      }

    val seed = 12345

    val featureSpaceSize = height * (paragraphVectors.getLayerSize + components.size() + products.size() + severityValues.size())

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

    val (_trainingData, _testData) = data.splitAt(9 * data.size / 10)

    val trainBatchSize = _trainingData.size / sc.defaultParallelism / 15

    // train data
    val trainRecordReader = new CollectionRecordReader(_trainingData)
    val trainIterator: DataSetIterator =
      new RecordReaderDataSetIterator(trainRecordReader, trainBatchSize, featureSpaceSize, possibleLabels)

    // test data
    val testRecordReader = new CollectionRecordReader(_testData)
    val testIterator: DataSetIterator =
      new RecordReaderDataSetIterator(testRecordReader, _testData.length, featureSpaceSize, possibleLabels)

    val saveUpdater = true

    //Execute training:
    val numEpochs = 2500

    //Perform evaluation (distributed)
    val testData = sc.parallelize(testIterator.toList)
    val trainingData = sc.parallelize(trainIterator.toList)

    val tm = new ParameterAveragingTrainingMaster.Builder(trainBatchSize)
      .rddTrainingApproach(RDDTrainingApproach.Direct)
      .averagingFrequency(averagingFrequency)
      .workerPrefetchNumBatches(2)
      .batchSizePerWorker(batchSize)
      .build()

    val networkListeners = new util.ArrayList[IterationListener]()
    networkListeners.add(new ScoreIterationListener(iterations / 5))

    val sparkNet = new SparkDl4jMultiLayer(sc, net, tm)
    sparkNet.setCollectTrainingStats(true)

    def statsStorage = new FileStatsStorage(new File("trainingStats.dl4j"));
    sparkNet.setListeners(statsStorage, networkListeners);

    log.info(s"Start training!!!")

    log.info("Epoch," +
      "Accuracy_Train,Precision_Train,Recall_Train,F1Score_Train,WPrecision_Train,WRecall_Train,WF1Score_Train," +
      "Accuracy_Test,Precision_Test,Recall_Test,F1Score_Test,WPrecision_Test,WRecall_Test,WF1Score_Test")

    val df: DecimalFormat = new DecimalFormat("0000")

    (startEpoch to numEpochs) foreach { i =>
      sparkNet.fit(trainingData)

      val locationToSave = new File(s"${architecture}_${inputFileName}_$i.zip")
      ModelSerializer.writeModel(net, locationToSave, saveUpdater)

      val evaluationTrain: SmartEvaluation = new SmartEvaluation(sparkNet.evaluate(trainingData))
      val evaluationTest: SmartEvaluation = new SmartEvaluation(sparkNet.evaluate(testData))

      log.info(s"${df.format(i)},${evaluationTrain.csvStats},${evaluationTest.csvStats}")
    }

    //Delete the temp training files, now that we are done with them
    tm.deleteTempFiles(sc)

  }

  /**
    * Fill info from Paragraph2Vec
    *
    * @param paragraphVectors
    * @param height
    * @param rowWithIndex
    * @return
    */
  private def addWord2VecPartitions(paragraphVectors: Broadcast[ParagraphVectors], height: Int, rowWithIndex: (util.List[Writable], Int)) = {
    addWord2Vec(paragraphVectors.value, height, rowWithIndex)
  }

  /**
    * Fill info from Paragraph2Vec
    *
    * @param paragraphVectors
    * @param height
    * @param rowWithIndex
    * @return
    */
  private def addWord2Vec(paragraphVectors: ParagraphVectors, height: Int, rowWithIndex: (util.List[Writable], Int)) = {
    val row = rowWithIndex._1
    val index = rowWithIndex._2
    if (index % 100 == 0) log.debug(s"Processed $index rows.")

    val desc = row.head.toString
    val words = desc.split("\\W+")

    val wordsPerGroup = 5

    val placeholder = Array.fill[DoubleWritable](paragraphVectors.getLayerSize)(new DoubleWritable(0)) ++ row.drop(1).dropRight(1)

    if (words.nonEmpty) {
      val groups = if (words.length < wordsPerGroup * height) {
        // if less than wordsPerGroup words in a group we pad the input
        val padded = (0 to (wordsPerGroup * height / words.length)).foldLeft[Array[String]](words) { (acc: Array[String], index: Int) => acc ++ words }
        val length = padded.length / height
        (0 until height) map (i => padded.slice(i * length, Math.min(padded.length, (i + 1) * length)))
      } else {
        val length = words.length / height
        (0 until height) map (i => words.slice(i * length, Math.min(words.length, (i + 1) * length)))
      }

      val vectors = groups map (slice =>
        Try(paragraphVectors.inferVector(slice.mkString(" ")).
          data().asDouble().map(new DoubleWritable(_)) ++ row.drop(1).dropRight(1)).
          getOrElse(placeholder)
        )
      seqAsJavaList(vectors.reduce(_ ++ _) ++ Seq(row.last))
    } else {
      val vectors = (0 until height) map (index => placeholder)
      seqAsJavaList(vectors.reduce(_ ++ _) ++ Seq(row.last))
    }
  }
}
