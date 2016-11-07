name := "mariana-triage"

version := "1.0"

packAutoSettings

scalaVersion := "2.11.8"

resolvers += "Local Maven Repository" at "file://" + "/home/aflorea" + "/.m2/repository"

libraryDependencies ++= Seq(
  "org.deeplearning4j" % "deeplearning4j-core" % "0.6.0"
  , "org.deeplearning4j" % "deeplearning4j-nlp" % "0.6.0" excludeAll ExclusionRule(organization = "ch.qos.logback")
  , "org.deeplearning4j" % "dl4j-spark_2.11" % "0.6.0"
  , "org.nd4j" % "nd4j-native-platform" % "0.6.0"
  , "org.datavec" % "datavec-api" % "0.6.0"
  , "org.datavec" %% "datavec-spark" % "0.6.0" excludeAll ExclusionRule(artifact = "slf4j-log4j12")

  , "com.beust" % "jcommander" % "1.7"
)

dependencyOverrides ++= Set(
  "com.fasterxml.jackson.core" % "jackson-databind" % "2.4.4"
)
