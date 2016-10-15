name := "mariana-triage"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq(
  "org.deeplearning4j" % "deeplearning4j-core" % "0.6.0"
  , "org.deeplearning4j" % "deeplearning4j-nlp" % "0.6.0"
  , "org.nd4j" % "nd4j-native-platform" % "0.6.0"
  , "org.datavec" % "datavec-api" % "0.6.0"
)