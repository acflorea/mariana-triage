package dr.acf.experiments

import java.io.File
import java.util
import java.util.{Collections, TimeZone}

import dr.acf.utils.SparkOps
import org.apache.spark.rdd.RDD
import org.datavec.api.records.reader.RecordReader
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.transform.TransformProcess
import org.datavec.api.transform.condition.ConditionOp
import org.datavec.api.transform.condition.column.IntegerColumnCondition
import org.datavec.api.transform.filter.ConditionFilter
import org.datavec.api.transform.schema.Schema
import org.datavec.api.util.ClassPathResource
import org.datavec.api.writable.{DoubleWritable, Writable}
import org.datavec.spark.transform.SparkTransformExecutor
import org.datavec.spark.transform.misc.StringToWritablesFunction
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{DenseLayer, GravesLSTM, OutputLayer, RnnOutputLayer}
import org.deeplearning4j.nn.conf._
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.plot.BarnesHutTsne
import org.deeplearning4j.text.documentiterator.{LabelAwareIterator, LabelledDocument}
import org.deeplearning4j.text.sentenceiterator.{BasicLineIterator, CollectionSentenceIterator}
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareListSentenceIterator
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.LoggerFactory

import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.util.Try

/**
  * Created by acflorea on 15/10/2016.
  */
object ParagraphVector extends SparkOps {

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
    .addColumnsString("original_text")
    .addColumnCategorical("bug_severity", severityValues)
    .addColumnInteger("component_id")
    .addColumnInteger("product_id")
    .addColumnInteger("class")
    .build()

  val log = LoggerFactory.getLogger(Classifier.getClass)

  def main(args: Array[String]): Unit = {

    System.out.println("OMP_NUM_THREADS " + System.getenv().get("OMP_NUM_THREADS"))

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
      //      .filter(new ConditionFilter(new IntegerColumnCondition("class", ConditionOp.GreaterOrEqual, 2)))
      // .filter(new ConditionFilter(new IntegerColumnCondition("component_id", ConditionOp.NotEqual, 128)))
      .removeAllColumnsExceptFor("text", "bug_severity", "component_id", "product_id", "class")
      .reorderColumns("text", "bug_severity", "component_id", "product_id", "class")
      .build()

    // After executing all of these operations, we have a new and different schema:
    val outputSchema: Schema = filterColumnsTransform.getFinalSchema

    log.info("\n\n\nSchema after transforming data:")
    log.info(s"$outputSchema")


    //=====================================================================
    //            Step 2.b: Transform
    //=====================================================================
    val directory: String = new ClassPathResource("netbeansbugs_filtered.csv").getFile.getPath
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

    //    val epochs = 5
    //    // ParagraphVectors training configuration
    //    val paragraphVectors = new ParagraphVectors.Builder()
    //      .learningRate(0.025).minLearningRate(0.001)
    //      .batchSize(2500).epochs(epochs)
    //      .iterate(ptvIterator).trainWordVectors(true)
    //      .tokenizerFactory(tokenizerFactory).build
    //
    //    // Start model training
    //    paragraphVectors.fit()
    //
    //    log.info("Save vectors....")
    //    WordVectorSerializer.writeParagraphVectors(paragraphVectors, "netbeans_05.model")

    val paragraphVectors = WordVectorSerializer.readParagraphVectors(new File("netbeans_05.model"))
    paragraphVectors.setTokenizerFactory(tokenizerFactory)


    //    log.info("Plot TSNE....");
    //    val tsne = new BarnesHutTsne.Builder()
    //      .setMaxIter(1000)
    //      .stopLyingIteration(250)
    //      .learningRate(500)
    //      .useAdaGrad(false)
    //      .theta(0.5)
    //      .setMomentum(0.5)
    //      .normalize(true)
    //      .build();
    //    paragraphVectors.lookupTable().plotVocab(tsne, paragraphVectors.getVocab.numWords(), new File("plot"))

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

    val possibleLabels = classes.size()

    val data = transformedData.collect().toList.collect { case row if row.head.toString.nonEmpty =>
      seqAsJavaList(paragraphVectors.inferVector(row.head.toString).
        data().asDouble().map(new DoubleWritable(_)) ++ row.drop(1))
    }

    //    val asJava: java.util.Collection[util.List[Writable]] = ListBuffer(data: _*)

    //    val outLibSVM = new ClassPathResource("netbeansbugs.libsvm").getFile
    //    // outLibSVM.createNewFile()
    //    val writer = new SVMLightRecordWriter(outLibSVM, true)
    //    data foreach {
    //      writer.write(_)
    //    }

    val featureSpaceSize = paragraphVectors.getLayerSize + components.size() + products.size() + severityValues.size()

    val (_trainingData, _testData) = data.splitAt(9 * data.size / 10)

    // train data
    val trainRecordReader = new CollectionRecordReader(_trainingData)
    val trainIterator: DataSetIterator =
      new RecordReaderDataSetIterator(trainRecordReader, Math.min(_trainingData.size, 100), featureSpaceSize, possibleLabels)

    // test data
    val testRecordReader = new CollectionRecordReader(_testData)
    val testIterator: DataSetIterator =
      new RecordReaderDataSetIterator(testRecordReader, _testData.length, featureSpaceSize, possibleLabels)

    //    val testRecordReader = new CollectionRecordReader(_trainingData)
    //    val testIterator: DataSetIterator =
    //      new RecordReaderDataSetIterator(testRecordReader, _trainingData.length, featureSpaceSize, possibleLabels)

    val numInputs = featureSpaceSize
    val outputNum = possibleLabels
    val iterations = 7500
    val seed = 6

    val h1size = 100
    val h2size = 100
    val h3size = 100
    val learningRate = 0.15
    val activation = "relu"
    val activation_end = "softmax"

    log.info("Build model....")
    //Set up network configuration
    val rnn_conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(iterations)
      .updater(Updater.RMSPROP)
      .regularization(true).l2(1e-5)
      .weightInit(WeightInit.XAVIER)
      .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
      .learningRate(0.0018)
      .list
      .layer(0, new GravesLSTM.Builder().nIn(featureSpaceSize).nOut(h1size).activation("softsign")
        .build())
      .layer(1, new RnnOutputLayer.Builder().activation("softmax")
        .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(h1size).nOut(outputNum)
        .build())
      .pretrain(false).backprop(true).build

    val tbpttLength = 50
    //Set up network configuration:
    val rnn_conf_2: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .learningRate(0.1)
      .rmsDecay(0.95)
      .seed(12345)
      .regularization(true).l2(0.001)
      .weightInit(WeightInit.XAVIER)
      .updater(Updater.RMSPROP)
      .list
      .layer(0, new GravesLSTM.Builder().nIn(numInputs).nOut(h1size).activation("tanh")
        .build())
      .layer(1, new GravesLSTM.Builder().nIn(h1size).nOut(h2size).activation("tanh")
        .build())
      .layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation("softmax")
        .nIn(h2size).nOut(outputNum)
        .build())
      .backpropType(BackpropType.TruncatedBPTT)
      .tBPTTForwardLength(tbpttLength)
      .tBPTTBackwardLength(tbpttLength)
      .pretrain(false).backprop(true).build

    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .activation(activation)
      .weightInit(WeightInit.XAVIER)
      .learningRate(learningRate)
      .regularization(true).l2(1e-4)
      .list
      .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(h1size)
        .build())
      .layer(1, new DenseLayer.Builder().nIn(h1size).nOut(h2size)
        .build())
      .layer(2, new DenseLayer.Builder().nIn(h2size).nOut(h3size)
        .build())
      .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .activation(activation_end)
        .nIn(h3size).nOut(outputNum).build())
      .backprop(true).pretrain(false)
      .build()

    //run the model
    val model: MultiLayerNetwork = new MultiLayerNetwork(rnn_conf_2)
    model.init()
    model.setListeners(new ScoreIterationListener(100))

    model.fit(trainIterator)

    val testData = testIterator.next()

    //evaluate the model on the test set
    val eval: Evaluation = new Evaluation(outputNum)
    //val output: INDArray = model.output(testData.getFeatures)
    //eval.eval(testData.getLabels, output)
    val output: INDArray = model.output(testData.getFeatures)
    eval.eval(testData.getLabels, output)
    log.info(eval.stats())

  }

}
