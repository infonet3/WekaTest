package wekatest;

import weka.core.Instances;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;

public class WekaTest {

    public static void main(String[] args) throws Exception {

        BufferedReader reader = new BufferedReader(new FileReader("/Users/mattjones/Desktop/MPGData.arff"));
        Instances trainData = new Instances(reader);
        reader.close();
 
        // setting class attribute
        //trainData.setClassIndex(trainData.numAttributes() - 1);
        trainData.setClassIndex(0);
    
        // train classifier
        Classifier cls = new LinearRegression();
        cls.buildClassifier(trainData);

        // evaluate classifier and print some statistics
        Evaluation eval = new Evaluation(trainData);
        eval.evaluateModel(cls, trainData);
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        
        //Now use the resulting model to predict the mpg
        Instances unlabeled = new Instances(new BufferedReader(new FileReader("/Users/mattjones/Desktop/MPGData.arff")));
        unlabeled.setClassIndex(0);
        Instances labeled = new Instances(unlabeled);
 
        // label instances
        for (int i = 0; i < unlabeled.numInstances(); i++) {
          double clsLabel = cls.classifyInstance(unlabeled.instance(i));
          labeled.instance(i).setClassValue(clsLabel);
        }

        // save labeled data
        BufferedWriter writer = new BufferedWriter(new FileWriter("/Users/mattjones/Desktop/MPGData-Labeled.arff"));
        writer.write(labeled.toString());
        writer.newLine();
        writer.flush();
        writer.close();
        
    }
   
}
