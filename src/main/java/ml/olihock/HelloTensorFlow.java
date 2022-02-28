/*
 * https://www.tensorflow.org/jvm/install
 */
package ml.olihock;

import org.tensorflow.*;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.math.Add;
import org.tensorflow.types.TInt32;

public class HelloTensorFlow {

    public static void main(String[] args) throws Exception {
        System.out.println("Hello TensorFlow " + TensorFlow.version());

        try (ConcreteFunction dbl = ConcreteFunction.create(HelloTensorFlow::dbl);
             TInt32 x = TInt32.scalarOf(10);
             Tensor dblX = dbl.call(x)) {
            System.out.println(x.getInt() + " doubled is " + ((TInt32)dblX).getInt());
        }

        Graph graph = new Graph();
    }

    private static Signature dbl(Ops tf) {
        Placeholder<TInt32> x = tf.placeholder(TInt32.class);
        Add<TInt32> dblX = tf.math.add(x, x);
        return Signature.builder().input("x", x).output("dbl", dblX).build();
    }

}