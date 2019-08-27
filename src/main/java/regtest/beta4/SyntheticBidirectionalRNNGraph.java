package regtest.beta4;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.CnnLossLayer;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.conf.layers.DepthwiseConvolution2D;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.Pooling2D;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SeparableConvolution2D;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.Upsampling2D;
import org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping2D;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class SyntheticBidirectionalRNNGraph {

    public static void main(String[] args) throws Exception {

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .l2(0.0001)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.005))
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.recurrent(10, 10))
                .addLayer("rnn1",
                        new Bidirectional(new LSTM.Builder()
                                .nOut(16)
                                .activation(Activation.RELU)
                                .build()), "input")
                .addLayer("rnn2",
                        new Bidirectional(new SimpleRnn.Builder()
                                .nOut(16)
                                .activation(Activation.RELU)
                                .build()), "input")
                .addVertex("concat", new MergeVertex(), "rnn1", "rnn2")
                .addLayer("pooling", new GlobalPoolingLayer.Builder()
                        .poolingType(PoolingType.MAX)
                        .poolingDimensions(2)
                        .collapseDimensions(true)
                        .build(), "concat")
                .addLayer("out", new OutputLayer.Builder().nOut(3).lossFunction(LossFunction.MCXENT).build(), "pooling")
                .setOutputs("out")
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        net.save(new File("output/SyntheticBidirectionalRNNGraph_100b4.bin"));
        Nd4j.getRandom().setSeed(12345);
        INDArray input = Nd4j.rand(new int[]{4, 10, 10});
        try (DataOutputStream dos = new DataOutputStream(
                new FileOutputStream(new File("output/SyntheticBidirectionalRNNGraph_Input_100b4.bin")))) {
            Nd4j.write(input, dos);
        }
        INDArray output = net.output(input)[0];
        try (DataOutputStream dos = new DataOutputStream(
                new FileOutputStream(new File("output/SyntheticBidirectionalRNNGraph_Output_100b4.bin")))) {
            Nd4j.write(output, dos);
        }
    }
}
