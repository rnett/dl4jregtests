/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package regtest.beta4;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Map;
import java.util.Random;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class HouseNumberDetection {
    private static final Logger log = LoggerFactory.getLogger(HouseNumberDetection.class);

    public static void main(String[] args) throws java.lang.Exception {
        // parameters matching the pretrained TinyYOLO model
        int width = 416;
        int height = 416;
        int nChannels = 3;
        int gridWidth = 13;
        int gridHeight = 13;

        // number classes (digits) for the SVHN datasets
        int nClasses = 10;

        // parameters for the Yolo2OutputLayer
        int nBoxes = 5;
        double lambdaNoObj = 0.5;
        double lambdaCoord = 1.0;
        double[][] priorBoxes = {{2, 5}, {2.5, 6}, {3, 7}, {3.5, 8}, {4, 9}};
        double detectionThreshold = 0.5;

        // parameters for the training phase
        int batchSize = 10;
        int nEpochs = 20;
        double learningRate = 1e-4;
        double lrMomentum = 0.9;

        int seed = 123;
        Random rng = new Random(seed);

//        SvhnDataFetcher fetcher = new SvhnDataFetcher();
//        File trainDir = fetcher.getDataSetPath(DataSetType.TRAIN);
//        File testDir = fetcher.getDataSetPath(DataSetType.TEST);


        log.info("Load data...");

//        FileSplit trainData = new FileSplit(trainDir, NativeImageLoader.ALLOWED_FORMATS, rng);
//        FileSplit testData = new FileSplit(testDir, NativeImageLoader.ALLOWED_FORMATS, rng);

        ObjectDetectionRecordReader recordReaderTrain = null; //new ObjectDetectionRecordReader(height, width, nChannels, gridHeight, gridWidth, new SvhnLabelProvider(trainDir));
//        recordReaderTrain.initialize(trainData);

        ObjectDetectionRecordReader recordReaderTest = null; //new ObjectDetectionRecordReader(height, width, nChannels, gridHeight, gridWidth, new SvhnLabelProvider(testDir));
//        recordReaderTest.initialize(testData);

        // ObjectDetectionRecordReader performs regression, so we need to specify it here
        RecordReaderDataSetIterator train = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, 1, true);
        train.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        RecordReaderDataSetIterator test = new RecordReaderDataSetIterator(recordReaderTest, 1, 1, 1, true);
        test.setPreProcessor(new ImagePreProcessingScaler(0, 1));


        ComputationGraph model;
        String modelFilename = "model.zip";

        if (new File(modelFilename).exists()) {
            log.info("Load model...");

            model = ModelSerializer.restoreComputationGraph(modelFilename);
        } else {
            log.info("Build model...");

            ComputationGraph pretrained = (ComputationGraph) TinyYOLO.builder().build().initPretrained();
            INDArray priors = Nd4j.create(priorBoxes).castTo(DataType.FLOAT);

            FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                    .seed(seed)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                    .gradientNormalizationThreshold(1.0)
                    .updater(new Adam.Builder().learningRate(learningRate).build())
                    //.updater(new Nesterovs.Builder().learningRate(learningRate).momentum(lrMomentum).build())
                    .l2(0.00001)
                    .activation(Activation.IDENTITY)
                    .trainingWorkspaceMode(WorkspaceMode.SEPARATE)
                    .inferenceWorkspaceMode(WorkspaceMode.SEPARATE)
                    .build();

            model = new TransferLearning.GraphBuilder(pretrained)
                    .fineTuneConfiguration(fineTuneConf)
                    .removeVertexKeepConnections("conv2d_9")
                    .removeVertexKeepConnections("outputs")
                    .addLayer("convolution2d_9",
                            new ConvolutionLayer.Builder(1, 1)
                                    .nIn(1024)
                                    .nOut(nBoxes * (5 + nClasses))
                                    .stride(1, 1)
                                    .convolutionMode(ConvolutionMode.Same)
                                    .weightInit(WeightInit.XAVIER)
                                    .activation(Activation.IDENTITY)
                                    .build(),
                            "leaky_re_lu_8")
                    .addLayer("outputs",
                            new Yolo2OutputLayer.Builder()
                                    .lambbaNoObj(lambdaNoObj)
                                    .lambdaCoord(lambdaCoord)
                                    .boundingBoxPriors(priors)
                                    .build(),
                            "convolution2d_9")
                    .setOutputs("outputs")
                    .build();

            model.save(new File("output/HouseNumberDetection_100b4.bin"));
            Nd4j.getRandom().setSeed(12345);
            INDArray input = Nd4j.rand(new int[]{3, 3, 416, 416});
            try (DataOutputStream dos = new DataOutputStream(
                    new FileOutputStream(new File("output/HouseNumberDetection_Input_100b4.bin")))) {
                Nd4j.write(input, dos);
            }
            INDArray output = model.outputSingle(input);
            try (DataOutputStream dos = new DataOutputStream(
                    new FileOutputStream(new File("output/HouseNumberDetection_Output_100b4.bin")))) {
                Nd4j.write(output, dos);
            }
        }
    }
}