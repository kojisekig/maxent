import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;

/**
 * Created with IntelliJ IDEA.
 * User: tpeng  <pengtaoo@gmail.com>
 * Date: 7/5/12
 * Time: 11:31 PM
 * To change this template use File | Settings | File Templates.
 */
public class MaxEntGD {
    // the number of training instances
    private final int N;

    // the minimal of Y
    private int minY;

    // the maximum of Y
    private int maxY;

    // the weight to learn.
    private double w[];

    private final List<Instance> instances;

    private List<FeatureFunction> functions;

    private List<Feature> features;

    public static void main(String... args) throws FileNotFoundException {
        List<Instance> instances = DataSet.readDataSet("examples/zoo.train");
        MaxEntGD me = new MaxEntGD(instances);
        me.train();

        List<Instance> trainInstances = DataSet.readDataSet("examples/zoo.test");
        int pass = 0;
        for (Instance instance : trainInstances) {
            int predict = me.classify(instance);
            if (predict == instance.getLabel()) {
                pass += 1;
            }
        }

        System.out.println("accuracy: " + 1.0 * pass / trainInstances.size());
    }

    public MaxEntGD(List<Instance> trainInstance) {

        instances = trainInstance;
        N = instances.size();
        createFeatFunctions(instances);
        w = new double[functions.size()];
    }

    private void createFeatFunctions(List<Instance> instances) {
        functions = new ArrayList<FeatureFunction>();
        features = new ArrayList<Feature>();

        int maxLabel = Integer.MIN_VALUE, minLabel = Integer.MAX_VALUE;
        int[] maxFeatures = new int[instances.get(0).getFeature().getValues().length];

        for (Instance instance : instances) {

            if (instance.getLabel() > maxLabel) {
              maxLabel = instance.getLabel();
            }
            if (instance.getLabel() < minLabel) {
              minLabel = instance.getLabel();
            }

            for (int i = 0; i < instance.getFeature().getValues().length; i++) {
                if (instance.getFeature().getValues()[i] > maxFeatures[i]) {
                    maxFeatures[i] = instance.getFeature().getValues()[i];
                }
            }

            features.add(instance.getFeature());
        }

        minY = minLabel;
        maxY = maxLabel;

        for (int i = 0; i < maxFeatures.length; i++) {
            for (int x = 0; x <= maxFeatures[i]; x++) {
                for (int y = minY; y <= maxY; y++) {
                    functions.add(new FeatureFunction(i, x, y));
                }
            }
        }

        System.out.println("# features = " + features.size());
        System.out.println("# functions = " + functions.size());
    }

    public void train() {
      System.out.println(Arrays.toString(w));

      // w で Gradient 求める
      final int K = 100;
      for(int i = 0; i < K; i++){
        w = updateByGradientDescent(w);
        System.out.println(Arrays.toString(w));
      }
    }

  static final double EPSILON = 0.1;

  double[] updateByGradientDescent(double[] w){
    double[] wu = gradient(w);
    for(int i = 0; i < wu.length; i++){
      w[i] = w[i] + EPSILON * wu[i];
    }
    return w;
  }

  double[] gradient(double[] w){
    double[] results = new double[w.length];
    for(Instance instance: instances){
      double[] f1 = getFeatureVector(instance);
      double[] f2 = f2(instance, w);
      results = sum(results, subtract(f1, f2));
    }
    return subtract(results, w); // C is 1.0
  }

  double z(double[] w, Instance instance){
    double result = 0;
    for(int label = minY; label <= maxY; label++){
      result += Math.exp(innerProduct(w, getFeatureVector(instance, label)));
    }
    return result;
  }

  static double innerProduct(double[] a, double[] b){
    assert(a != null && b != null && a.length == b.length);
    double ip = 0;
    for(int i = 0; i < a.length; i++){
      ip += a[i] * b[i];
    }
    return ip;
  }

  double[] getFeatureVector(Instance instance){
    double[] fv = new double[functions.size()];
    int i = 0;
    for(FeatureFunction featureFunction: functions){
      fv[i++] = featureFunction.apply(instance.getFeature(), instance.getLabel());
    }
    return fv;
  }

  double[] getFeatureVector(Instance instance, int label){
    double[] fv = new double[functions.size()];
    int i = 0;
    for(FeatureFunction featureFunction: functions){
      fv[i++] = featureFunction.apply(instance.getFeature(), label);
    }
    return fv;
  }

  double[] f2(Instance instance, double[] w){
    double z = z(w, instance);
    double[] results = new double[functions.size()];

    for(int label = minY; label <= maxY; label++){
      double[] ff = getFeatureVector(instance, label);
      results = sum(results, times(ff, Math.exp(innerProduct(w, ff))));
    }

    return times(results, 1.0 / z);
  }

  static double[] subtract(double[] a, double[] b){
    assert(a != null && b != null && a.length == b.length);
    double[] results = new double[a.length];
    for(int i = 0; i < a.length; i++){
      results[i] = a[i] - b[i];
    }
    return results;
  }

  static double[] sum(double[] a, double[] b){
    assert(a != null && b != null && a.length == b.length);
    double[] results = new double[a.length];
    for(int i = 0; i < a.length; i++){
      results[i] = a[i] + b[i];
    }
    return results;
  }

  static double[] times(double[] a, double b){
    double[] results = new double[a.length];
    for(int i = 0; i < a.length; i++){
      results[i] = a[i] * b;
    }
    return results;
  }

    public int classify(Instance instance) {

        double max = 0;
        int label = 0;

        for (int y = minY; y <= maxY; y++) {
            double sum = 0;
            for (int i = 0; i < functions.size(); i++) {
                sum += Math.exp(w[i] * functions.get(i).apply(instance.getFeature(), y));
            }
            if (sum > max) {
                max = sum;
                label = y;
            }
        }
        return label;
    }

    static class FeatureFunction {

        private int index;
        private int value;
        private int label;

        FeatureFunction(int index, int value, int label) {
            this.index = index;
            this.value = value;
            this.label = label;
        }

        public int apply(Feature feature, int label) {
            if (feature.getValues()[index] == value && label == this.label)
                return 1;
            return 0;
        }
    }
}


