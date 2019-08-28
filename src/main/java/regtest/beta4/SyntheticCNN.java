package regtest.beta4;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.CnnLossLayer;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.conf.layers.DepthwiseConvolution2D;
import org.deeplearning4j.nn.conf.layers.Pooling2D;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SeparableConvolution2D;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.Upsampling2D;
import org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping2D;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class SyntheticCNN {

    public static void main(String[] args) throws Exception {
        int height = 28;    // height of the picture in px
        int width = 28;     // width of the picture in px
        int channels = 1;   // single channel for grayscale images

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .l2(0.0001)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.005))
                .list()
                .layer(new Convolution2D.Builder()
                        .kernelSize(3, 3)
                        .stride(2, 1)
                        .nOut(4).nIn(channels)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SeparableConvolution2D.Builder()
                        .kernelSize(3, 3)
                        .convolutionMode(ConvolutionMode.Same)
                        .nOut(8)
                        .activation(Activation.RELU)
                        .build())
                .layer(new Pooling2D.Builder().kernelSize(3, 3).poolingType(PoolingType.MAX).build())
                .layer(new ZeroPaddingLayer(4, 4))
                .layer(new Upsampling2D.Builder().size(3).build())
                .layer(new DepthwiseConvolution2D.Builder()
                        .kernelSize(3, 3)
                        .depthMultiplier(2)
                        .nOut(16)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder().kernelSize(2, 2).poolingType(PoolingType.MAX).build())
                .layer(new Cropping2D.Builder(3, 2).build())
                .layer(new Convolution2D.Builder().kernelSize(4, 4).nOut(4).build())
                .layer(new CnnLossLayer.Builder().lossFunction(LossFunction.MEAN_ABSOLUTE_ERROR).build())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.save(new File("output/SyntheticCNN_100b4.bin"));
        Nd4j.getRandom().setSeed(12345);
        INDArray input = Nd4j.rand(new int[]{4, 1, 28, 28});
        try (DataOutputStream dos = new DataOutputStream(
                new FileOutputStream(new File("output/SyntheticCNN_Input_100b4.bin")))) {
            Nd4j.write(input, dos);
        }
        INDArray output = net.output(input);
        try (DataOutputStream dos = new DataOutputStream(
                new FileOutputStream(new File("output/SyntheticCNN_Output_100b4.bin")))) {
            Nd4j.write(output, dos);
        }
    }
}
