package com.example.ondevice_sample;

import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;

import org.tensorflow.lite.examples.transfer.api.TransferLearningModel;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {
    private TransferLearningModelWrapper tlModel;
    private  Bitmap bitmap;

    private static final int MAX_RESULTS = 3;
    private static final int BATCH_SIZE = 1;
    private static final int PIXEL_SIZE = 3;
    private static final float THRESHOLD = 0.1f;

    private static final int INPUT_SIZE = 224;
    private static final int CROP_SIZE = 224;
    private static final String IMG_FILE = "sampleImg.jpg";
    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128.0f;

    private static final float IMAGE_STD2 = 127.5f;
    private static final float IMAGE_MEAN2 = 1.0f;
    private static final int BYTE_SIZE_OF_FLOAT = 4;
    private static final int LOWER_BYTE_MASK = 0xFF;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        tlModel = new TransferLearningModelWrapper(getApplicationContext());

        bitmap= assetsRead(IMG_FILE);
        //bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, false);
        float[][][] rgbImage = prepareImage(bitmap , 224, 224);


        Button button = (Button) findViewById(R.id.button_infer);

        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                TransferLearningModel.Prediction[] predictions = tlModel.predict(rgbImage);
                if (predictions == null) {
                    return;
                }else {
                    Log.i("JunDebug", "predictions " + predictions.getClass());
                }
            }
        });


    }

    public Bitmap assetsRead(String file) {

        InputStream is;
        Bitmap bitmap = null;

        try {

            AssetManager assetManager = getResources().getAssets();
            is = assetManager.open(file);

            int size = is.available();
            byte[] buffer = new byte[size];
            is.read(buffer);
            is.close();
            bitmap = BitmapFactory.decodeByteArray( buffer, 0, buffer.length ) ;
            bitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true); }
        catch (IOException e) {
            e.printStackTrace();
        }

        return bitmap ;
    }


    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap, int W, int H) {

        ByteBuffer byteBuffer;
        byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * H * W * PIXEL_SIZE);
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] intValues = new int[W * H];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        int pixel = 0;
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j <  W; ++j) {

                final int val = intValues[pixel++];
                byteBuffer.putFloat((((val >> 16) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                byteBuffer.putFloat((((val >> 8) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                byteBuffer.putFloat((((val) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
            }
        }

        return byteBuffer;
    }
    private static float [][][] prepareImage(Bitmap bitmap, int IMG_WIDTH, int IMG_HEIGHT){
        int modelImageSize = IMG_WIDTH;

        float[][][] normalizedRgb = new float[modelImageSize][modelImageSize][3];
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(
                bitmap, modelImageSize, modelImageSize, true);

        for (int y = 0; y < modelImageSize; y++) {
            for (int x = 0; x < modelImageSize; x++) {
                int rgb = scaledBitmap.getPixel(x, y);

                float r = ((rgb >> 16) & LOWER_BYTE_MASK) * (1 / 255.f);
                float g = ((rgb >> 8) & LOWER_BYTE_MASK) * (1 / 255.f);
                float b = (rgb & LOWER_BYTE_MASK) * (1 / 255.f);

                normalizedRgb[y][x][0] = r;
                normalizedRgb[y][x][1] = g;
                normalizedRgb[y][x][2] = b;
            }
        }

        return normalizedRgb;
    }
}


