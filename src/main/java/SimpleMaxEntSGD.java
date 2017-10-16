/*
 * 「言語処理のための機械学習入門」より。
 */

import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

public class SimpleMaxEntSGD {

  private static final String[] DOCS = {
      "good bad good good",
      "exciting exciting",
      "bad boring boring boring",
      "bad exciting bad"
  };

  private static final Person[] DOC_PERSON = {
      Person.P, Person.P, Person.N, Person.N
  };

  private static final String[] WORDS = {
      "good", "bad", "exciting", "boring"
  };

  // 各 doc が保有する単語集合
  private static Set<String>[] DOC_WORDS;

  public static void main(String[] args){
    // DOC_WORDS の初期化
    DOC_WORDS = new Set[DOCS.length];
    for(int i = 0; i < DOCS.length; i++){
      DOC_WORDS[i] = getWordSet(DOCS[i]);
    }

    // w の初期値
    double[] w = new double[]{ 0, 0, 0, 0, 0, 0, 0, 0 };
    System.out.println(Arrays.toString(w));

    long st = System.currentTimeMillis();

    // w で Gradient 求める
    final int K = 10;
    for(int i = 0; i < K; i++){
      w = updateByStochasticGradientDescent(w);
      System.out.println(Arrays.toString(w));
    }

    long et = System.currentTimeMillis();

    System.out.printf("\n%d msec\n", et - st);

    // モデルのテスト
    test("exciting boring", w);
    test("good exciting boring", w);
    test("bad exciting boring", w);
  }

  static void test(String doc, double[] w){
    double[] fp = getFeatureVector(doc, Person.P);
    double[] fn = getFeatureVector(doc, Person.N);

    double pp = innerProduct(w, fp);
    double pn = innerProduct(w, fn);
    if(pp > pn){
      System.out.printf("\n\"%s\" => %f > %f => P", doc, pp, pn);
    }
    else{
      System.out.printf("\n\"%s\" => %f < %f => N", doc, pp, pn);
    }
  }

  static Set<String> getWordSet(String doc){
    Set<String> wordSet = new HashSet<String>();
    String[] words = doc.split(" ");
    for(String word: words){
      wordSet.add(word);
    }
    return wordSet;
  }

  static final double EPSILON = 0.1;

  static double[] updateByStochasticGradientDescent(double[] w){
    shuffle();
    double[] wu = new double[w.length];
    System.arraycopy(w, 0, wu, 0, w.length);
    for(int docId = 0; docId < DOCS.length; docId++){
      wu = gradient(docId, wu);
    }
    for(int i = 0; i < wu.length; i++){
      w[i] = w[i] + EPSILON * wu[i];
    }
    return w;
  }

  private static Random r = new Random(System.currentTimeMillis());

  static void shuffle(){
    for(int i = 0; i < 100; i++){
      int j = r.nextInt(DOCS.length);
      int k = r.nextInt(DOCS.length);
      if(j != k){
        // do swap
        Person ap = DOC_PERSON[j];
        DOC_PERSON[j] = DOC_PERSON[k];
        DOC_PERSON[k] = ap;
        Set<String> aw = DOC_WORDS[j];
        DOC_WORDS[j] = DOC_WORDS[k];
        DOC_WORDS[k] = aw;
      }
    }
  }

  static double[] gradient(int docId, double[] w){
    double[] f1 = getFeatureVector(docId, DOC_PERSON[docId]);
    double[] f2 = f2(docId, w);
    return subtract(subtract(f1, f2), w);   // C is 1.0
  }

  static double z(double[] w, int docId){
    double result = 0;
    result += Math.exp(innerProduct(w, getFeatureVector(docId, Person.P)));
    result += Math.exp(innerProduct(w, getFeatureVector(docId, Person.N)));
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

  static double[] getFeatureVector(int docId, Person person){
    return getFeatureVector(DOC_WORDS[docId], person);
  }

  static double[] getFeatureVector(String doc, Person person){
    return getFeatureVector(getWordSet(doc), person);
  }

  // 特徴ベクトルの次元数 = 単語数 ｘ クラス数
  // { good, bad, exciting, boring } x { P, N }
  // => good-P, bad-P, exciting-P, boring-P,
  //    good-N, bad-N, exciting-N, boring-N
  static double[] getFeatureVector(Set<String> doc, Person person){
    double[] fv = new double[WORDS.length * Person.values().length];
    int i = 0;
    for(String word: WORDS){
      fv[i++] = doc.contains(word) && person.equals(Person.P) ? 1 : 0;
    }
    for(String word: WORDS){
      fv[i++] = doc.contains(word) && person.equals(Person.N) ? 1 : 0;
    }
    return fv;
  }

  static double[] f2(int docId, double[] w){
    double z = z(w, docId);

    double[] ffp = getFeatureVector(docId, Person.P);
    double[] results = times(ffp, Math.exp(innerProduct(w, ffp)));

    double[] ffn = getFeatureVector(docId, Person.N);
    results = sum(results, times(ffn, Math.exp(innerProduct(w, ffn))));

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

  static enum Person {
    P, N;
  }
}
