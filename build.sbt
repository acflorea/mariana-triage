name := "mariana-triage"

version := "1.1"

packAutoSettings

scalaVersion := "2.11.8"

val dl4jVersion = "0.7.1"

libraryDependencies ++= Seq(
  "org.deeplearning4j" % "deeplearning4j-core" % dl4jVersion
  , "org.deeplearning4j" % "deeplearning4j-nlp" % dl4jVersion excludeAll ExclusionRule(organization = "ch.qos.logback")
  , "org.deeplearning4j" % "dl4j-spark_2.11" % dl4jVersion
  , "org.nd4j" % "nd4j-native-platform" % dl4jVersion
  , "org.nd4j" % "nd4j-native" % dl4jVersion
  , "org.datavec" % "datavec-api" % dl4jVersion
  , "org.datavec" %% "datavec-spark" % dl4jVersion

  , "com.beust" % "jcommander" % "1.7"
).map (_ excludeAll ExclusionRule (artifact = "slf4j-log4j12"))

dependencyOverrides ++= Set(
   // "com.fasterxml.jackson.core" % "jackson-databind" % "2.4.4"
)
