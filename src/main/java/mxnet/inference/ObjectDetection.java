package mxnet.inference;

import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * An example of inference using an object detection model.
 *
 * <p>See this <a
 * href="https://github.com/deepjavalibrary/djl/blob/master/examples/docs/object_detection.md">doc</a>
 * for information about this example.
 */
public final class ObjectDetection {

    private static final String IMAGE_DIR = "src/main/resources/";
    private static final String OUTPUT_IMAGE_DIR = "build/output";
    private static final String IMAGE_NAME = "dog.jpg";//kitten.jpg,largest_selfie.jpg
    private static final String DETECTED_IMAGE_NAME = "detected-" + IMAGE_NAME;//kitten.jpg,largest_selfie.jpg

    private ObjectDetection() {}

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        DetectedObjects detection = ObjectDetection.predict();
        System.out.print(detection);
    }

    public static DetectedObjects predict() throws IOException, ModelException, TranslateException {

        Path imageFile = Paths.get(IMAGE_DIR + IMAGE_NAME);
        Image img = ImageFactory.getInstance().fromFile(imageFile);

        String backbone;
        if ("TensorFlow".equals(Engine.getInstance().getEngineName())) {
            backbone = "mobilenet_v2";
        } else {
            backbone = "resnet50";
        }

        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(Image.class, DetectedObjects.class)
                        .optFilter("backbone", backbone)
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<Image, DetectedObjects> model = ModelZoo.loadModel(criteria)) {
            try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
                DetectedObjects detection = predictor.predict(img);
                saveBoundingBoxImage(img, detection);
                return detection;
            }
        }
    }

    private static void saveBoundingBoxImage(Image img, DetectedObjects detection)
            throws IOException {
        Path outputDir = Paths.get(OUTPUT_IMAGE_DIR);
        Files.createDirectories(outputDir);

        // Make image copy with alpha channel because original image was jpg
        Image newImage = img.duplicate(Image.Type.TYPE_INT_ARGB);
        newImage.drawBoundingBoxes(detection);

        Path imagePath = outputDir.resolve(DETECTED_IMAGE_NAME);
        // OpenJDK can't save jpg with alpha channel
        newImage.save(Files.newOutputStream(imagePath), "png");
        System.out.printf("Detected objects image has been saved in: %s", imagePath);
    }
}
