package dr.acf.experiments

import java.io.File
import java.util
import java.util.TimeZone

import com.beust.jcommander.{JCommander, Parameter}
import dr.acf.utils.{SmartEvaluation, SparkOps, WordVectorSmartSerializer}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.datavec.api.records.reader.RecordReader
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.transform.TransformProcess
import org.datavec.api.transform.schema.Schema
import org.datavec.api.writable.{DoubleWritable, Writable}
import org.datavec.spark.transform.SparkTransformExecutor
import org.datavec.spark.transform.misc.StringToWritablesFunction
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors
import org.deeplearning4j.models.sequencevectors.SequenceVectors
import org.deeplearning4j.models.sequencevectors.enums.ListenerEvent
import org.deeplearning4j.models.sequencevectors.interfaces.VectorsListener
import org.deeplearning4j.models.word2vec.VocabWord
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf._
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.RBM.{HiddenUnit, VisibleUnit}
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory

import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.util.Try

/**
  * Created by acflorea on 15/10/2016.
  */
object ParagraphVector extends SparkOps {

  object Args {
    // Declared as var because JCommander assigns a new collection declared
    // as java.util.List because that's what JCommander will replace it with.
    // It'd be nice if JCommander would just use the provided List so this
    // could be a val and a Scala LinkedList.
    @Parameter(
      names = Array("-rf", "--resourceFolder"),
      description = "Location of resource files.")
    var resourceFolder: String = "/Uers/aflorea/phd/mariana-triage/data"

    @Parameter(
      names = Array("-f", "--inputFile"),
      description = "Input File name")
    val inputFileName = "netbeansbugs_filtered.csv"

    @Parameter(
      names = Array("-ce", "--computeEmbeddings"),
      description = "Weather to compute the embeddings or attempt to load an existing model")
    val computeEmbeddings = false

    @Parameter(
      names = Array("-ep", "--epochs"),
      description = "Number of epochs for embeddings")
    val epochsForEmbeddings = 20

    @Parameter(
      names = Array("-m", "--model"),
      description = "Name of model file to user")
    val model = "netbeans"

    @Parameter(
      names = Array("-a", "--architecture"),
      description = "Type of the architecture to use")
    val architecture = "rnn" // rnn, cnn, deep

  }

  lazy val modelName = if (Args.epochsForEmbeddings < 10)
    s"${Args.model}_0${Args.epochsForEmbeddings}.model"
  else
    s"${Args.model}_${Args.epochsForEmbeddings}.model"

  val severityValues = util.Arrays.asList("normal", "enhancement", "major", "trivial", "critical", "minor", "blocker")
  val statusValues = util.Arrays.asList("CLOSED", "RESOLVED", "VERIFIED", "trivial", "critical", "minor")

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

  val log = LoggerFactory.getLogger(Classifier.getClass)

  def main(args: Array[String]): Unit = {

    // Initialize jCommander
    new JCommander(Args, args.toArray: _*)

    log.debug(s"Resource folder is ${Args.resourceFolder}")
    log.debug(s"Input file is ${Args.inputFileName}")
    log.debug(s"Compute embeddings is ${Args.computeEmbeddings}")
    log.debug(s"Number of embedding epochs is ${Args.epochsForEmbeddings}")

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
      //            .filter(new ConditionFilter(new IntegerColumnCondition("class", ConditionOp.GreaterOrEqual, 2)))
      //.filter(new ConditionFilter(new IntegerColumnCondition("component_id", ConditionOp.NotEqual, 128)))
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
    val directory: String = Args.resourceFolder + Args.inputFileName
    val stringData: RDD[String] = sc.textFile(directory)

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

    distinctClasses.zipWithIndex map { classWithIndex =>
      classes.put(classWithIndex._1, classWithIndex._2.toString)
    }

    //=====================================================================
    //            PARAGRAPH VECTOR !!!
    //=====================================================================

    // build a iterator for our dataset
    val ptvIterator = new CollectionSentenceIterator(descs.flatMap(_.split("\\. ")).toList)
    //
    val tokenizerFactory = new DefaultTokenizerFactory
    tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor)

    // Lister to store model at each phase
    val vectorListener = new VectorsListener[VocabWord] {
      override def processEvent(event: ListenerEvent, sequenceVectors: SequenceVectors[VocabWord], argument: Long) = {
        event match {
          case ListenerEvent.EPOCH if argument % 3 == 0 =>
            log.info("Save vectors....")
            lazy val _modelName = if (argument < 10)
              s"${Args.model}_0$argument.model"
            else
              s"${Args.model}_$argument.model"
            WordVectorSmartSerializer
              .writeParagraphVectors(sequenceVectors.asInstanceOf[ParagraphVectors], Args.resourceFolder + modelName)
          case _ =>
        }
      }

      override def validateEvent(event: ListenerEvent, argument: Long): Boolean = true
    }

    val paragraphVectors = if (Args.computeEmbeddings) {

      log.info("Build Embedded Vectors ....")
      // ParagraphVectors training configuration
      val _paragraphVectors = new ParagraphVectors.Builder()
        .setVectorsListeners(new util.ArrayList[VectorsListener[VocabWord]] {
          vectorListener
        })
        .layerSize(50)
        .learningRate(0.025).minLearningRate(0.001)
        .batchSize(2500).epochs(Args.epochsForEmbeddings)
        .iterate(ptvIterator).trainWordVectors(true)
        .tokenizerFactory(tokenizerFactory).build

      // Start model training
      _paragraphVectors.fit()

      log.info("Save vectors....")
      WordVectorSmartSerializer.writeParagraphVectors(_paragraphVectors, Args.resourceFolder + modelName)
      _paragraphVectors

    } else {

      log.info("Load Embedded Vectors ....")
      val _paragraphVectors = WordVectorSmartSerializer.readParagraphVectors(new File(Args.resourceFolder + modelName))
      _paragraphVectors.setTokenizerFactory(tokenizerFactory)
      _paragraphVectors

    }

    log.info("Embedded Vectors OK ....")

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

    val height = Args.architecture match {
      case "cnn" => 10
      case "rnn" => 1
      case "deep" => 1
    }

    log.info("Apply Paragraph2Vec ....")

    val broadcastPV = false
    val data =
      if (broadcastPV) {
        val pv = sc.broadcast(paragraphVectors)
        transformedData.mapPartitions {
          _ map {
            case row if row.head.toString.nonEmpty =>
              seqAsJavaList(pv.value.inferVector(row.head.toString).
                data().asDouble().map(new DoubleWritable(_)) ++ row.drop(1))
          }
        }.collect().toList
      } else {
        transformedData.collect().zipWithIndex.toList.collect { case rowWithIndex if rowWithIndex._1.head.toString.nonEmpty =>
          val row = rowWithIndex._1
          val index = rowWithIndex._2
          if (index % 100 == 0) log.debug(s"Processed $index rows.")

          val desc = row.head.toString
          val words = desc.split("\\W+")

          val groups = if (words.length < 3 * height) {
            // if less than 3 words in a group we pad the input
            val padded = (0 to (3 * height / words.length)).foldLeft[Array[String]](words) { (acc: Array[String], index: Int) => acc ++ words }
            val length = padded.length / height
            (0 until height) map (i => padded.slice(i * length, Math.min(padded.length, (i + 1) * length)))
          } else {
            val length = words.length / height
            (0 until height) map (i => words.slice(i * length, Math.min(words.length, (i + 1) * length)))
          }

          val vectors = (groups map (slice => paragraphVectors.inferVector(slice.mkString(" ")).
            data().asDouble().map(new DoubleWritable(_)) ++ row.drop(1).dropRight(1)))
          seqAsJavaList(vectors.reduce(_ ++ _) ++ Seq(row.last))
        }
      }

    val featureSpaceSize = height * (paragraphVectors.getLayerSize + components.size() + products.size() + severityValues.size())

    val batchSize = 100
    val averagingFrequency = 5

    val outputNum = possibleLabels
    val iterations = 500

    val layer1width = 100
    val learningRate = 0.0018
    val activation = "softsign"

    val activation_end = "softmax"

    val (_trainingData, _testData) = data.splitAt(9 * data.size / 10)

    // train data
    val trainRecordReader = new CollectionRecordReader(_trainingData)
    val trainIterator: DataSetIterator =
      new RecordReaderDataSetIterator(trainRecordReader, Math.min(_trainingData.size, batchSize), featureSpaceSize, possibleLabels)

    // test data
    val testRecordReader = new CollectionRecordReader(_testData)
    val testIterator: DataSetIterator =
      new RecordReaderDataSetIterator(testRecordReader, _testData.length, featureSpaceSize, possibleLabels)

    log.info("Build model....")
    log.info(s"Number of iterations $iterations")
    log.info(s"Number of features $featureSpaceSize")

    val seed = 12345

    val cnn_conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .seed(12345).iterations(iterations).regularization(true).l2(0.0005).learningRate(.01)
      .weightInit(WeightInit.XAVIER).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(Updater.NESTEROVS).momentum(0.9)
      .list
      .layer(0, new ConvolutionLayer.Builder(2, featureSpaceSize).name("conv1")
        // nIn is the number of channels, nOut is the number of filters to be applied
        .nIn(1).stride(1, 1).nOut(20)
        .activation("identity").build())
      .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG).name("pooling_1")
        .kernelSize(1, featureSpaceSize).stride(1, 1).build())
      .layer(2, new ConvolutionLayer.Builder(1, featureSpaceSize).name("conv2")
        .stride(1, 1).nOut(50).activation("identity").build())
      .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG).name("pooling_2")
        .kernelSize(1, featureSpaceSize).stride(1, 1).build())
      .layer(4, new DenseLayer.Builder().activation("relu").nOut(500).build())
      .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(outputNum).activation("softmax").build())
      // height, width, height
      .setInputType(InputType.convolutionalFlat(height, featureSpaceSize / height, 1))
      .backprop(true).pretrain(false).build()

    //Set up network configuration
    val rnn_conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
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
      .pretrain(false).backprop(true).build


    val deep_conf = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .optimizationAlgo(OptimizationAlgorithm.LBFGS)
      .list()
      .layer(0, new RBM.Builder().l2(1e-1).l1(1e-3)
        .nIn(featureSpaceSize) // Input nodes
        .nOut(layer1width) // Output nodes
        .activation("relu") // Activation function type
        .weightInit(WeightInit.RELU) // Weight initialization
        .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).k(3)
        .hiddenUnit(HiddenUnit.RECTIFIED).visibleUnit(VisibleUnit.GAUSSIAN)
        .updater(Updater.ADAGRAD).gradientNormalization(GradientNormalization.ClipL2PerLayer)
        .build())
      .layer(1, new RBM.Builder().l2(1e-1).l1(1e-3)
        .nIn(layer1width) // Input nodes
        .nOut(layer1width) // Output nodes
        .activation("relu") // Activation function type
        .weightInit(WeightInit.RELU) // Weight initialization
        .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).k(3)
        .hiddenUnit(HiddenUnit.RECTIFIED).visibleUnit(VisibleUnit.GAUSSIAN)
        .updater(Updater.ADAGRAD).gradientNormalization(GradientNormalization.ClipL2PerLayer)
        .build())
      .layer(2, new RBM.Builder().l2(1e-1).l1(1e-3)
        .nIn(layer1width) // Input nodes
        .nOut(layer1width) // Output nodes
        .activation("relu") // Activation function type
        .weightInit(WeightInit.RELU) // Weight initialization
        .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).k(3)
        .hiddenUnit(HiddenUnit.RECTIFIED).visibleUnit(VisibleUnit.GAUSSIAN)
        .updater(Updater.ADAGRAD).gradientNormalization(GradientNormalization.ClipL2PerLayer)
        .build())
      .layer(3, new RBM.Builder().l2(1e-1).l1(1e-3)
        .nIn(layer1width) // Input nodes
        .nOut(layer1width) // Output nodes
        .activation("relu") // Activation function type
        .weightInit(WeightInit.RELU) // Weight initialization
        .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).k(3)
        .hiddenUnit(HiddenUnit.RECTIFIED).visibleUnit(VisibleUnit.GAUSSIAN)
        .updater(Updater.ADAGRAD).gradientNormalization(GradientNormalization.ClipL2PerLayer)
        .build())
      .layer(4, new RBM.Builder().l2(1e-1).l1(1e-3)
        .nIn(layer1width) // Input nodes
        .nOut(layer1width) // Output nodes
        .activation("relu") // Activation function type
        .weightInit(WeightInit.RELU) // Weight initialization
        .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).k(3)
        .hiddenUnit(HiddenUnit.RECTIFIED).visibleUnit(VisibleUnit.GAUSSIAN)
        .updater(Updater.ADAGRAD).gradientNormalization(GradientNormalization.ClipL2PerLayer)
        .build())
      .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation("softmax").nIn(layer1width).nOut(outputNum).build())
      .pretrain(false).backprop(true)
      .build()

    log.info(s"Architecture ${Args.architecture}")
    val active_conf = Args.architecture match {
      case "cnn" => cnn_conf
      case "rnn" => rnn_conf
      case "deep" => deep_conf
    }

    log.info(s"Network configuration ${active_conf.toString}")

    val tm = new ParameterAveragingTrainingMaster.Builder(1)
      .averagingFrequency(averagingFrequency)
      .workerPrefetchNumBatches(2)
      .batchSizePerWorker(batchSize)
      .build()

    //Create the Spark network
    val sparkNet = new SparkDl4jMultiLayer(sc, active_conf, tm)

    //Execute training:
    val numEpochs = 150

    //Perform evaluation (distributed)
    val testData = sc.parallelize(testIterator.toList).persist(StorageLevel.MEMORY_AND_DISK_SER)
    val trainingData = sc.parallelize(trainIterator.toList).persist(StorageLevel.MEMORY_AND_DISK_SER)

    log.info(s"Start training!!!")

    (1 to numEpochs) foreach { i =>
      sparkNet.fit(trainingData)
      log.info(s"Completed Epoch $i")

      val evaluationTrain: SmartEvaluation = new SmartEvaluation(sparkNet.evaluate(trainingData))
      log.info("***** Evaluation TRAIN DATA *****")
      log.info(evaluationTrain.stats(false, false))

      val evaluationTest: SmartEvaluation = new SmartEvaluation(sparkNet.evaluate(testData))
      log.info("***** Evaluation TEST DATA *****")
      log.info(evaluationTest.stats(false, false))
    }

    //Delete the temp training files, now that we are done with them
    tm.deleteTempFiles(sc)

  }

}
