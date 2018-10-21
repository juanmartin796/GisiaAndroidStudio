package gisia.martin.com.perceptron;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.TextView;
import android.widget.Toast;

public class MainActivity extends AppCompatActivity {
    TextView tvResultado;

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Example of a call to a native method
        tvResultado = (TextView) findViewById(R.id.tvResultado);
        convolucionModel();
        pool();
        //float result = modelTraining();
        float result = 2;
        tvResultado.setText("El resultado es: "+ String.valueOf(result));
        Toast.makeText(this, "Resultado "+ result, Toast.LENGTH_LONG).show();
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();

    public native float modelTraining();

    public native void convolucionModel();

    public native void pool();
}
