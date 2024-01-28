package com.example.gamejadul;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.text.Html;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {
    Button buttonGalery, buttonKamera, buttonDeteksi;
    TextView ResultText, ClassifiedText;
    ImageView imageView;
    Bitmap bitmap;
    private Interpreter tflite;
    private String[] LABELS = {"Bike", "Car"};
    private String modelPath = "ml/carvsbikevgg16.tflite";
    private final int NUM_CLASSES = 7; // Masukkan jumlah kelas pada model Anda;
    private final int INPUT_SIZE = 224; // Sesuaikan dengan ukuran input model Anda
    static final int REQUEST_IMAGE_CAPTURE = 1;
    static final int CAMERA_PERMISSION_CODE = 100;

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        buttonDeteksi = findViewById(R.id.buttonDeteksi);
        buttonGalery = findViewById(R.id.buttonGalery);
        buttonKamera = findViewById(R.id.buttonKamera);
        ResultText = findViewById(R.id.ResultText);
        ClassifiedText = findViewById(R.id.ClassifiedText);
        imageView  = findViewById(R.id.imageView);

        try {
            tflite = new Interpreter(loadModelFile(getApplicationContext().getAssets(), "ml/carvsbikevgg16.tflite"));
        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(this, "Gagal memuat model", Toast.LENGTH_SHORT).show();
            finish();
        }

        buttonKamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                checkCameraPermission();
            }
        });

        buttonGalery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent();
                intent.setAction(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent, 10);
            }
        });

        buttonDeteksi.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (bitmap != null) {
                    runInference();
                } else {
                    Toast.makeText(MainActivity.this, "Gambar belum diambil", Toast.LENGTH_SHORT).show();
                }
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode==10){
            if (data!=null){
                Uri uri = data.getData();
                try {
                    bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
                    imageView.setImageBitmap(bitmap);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        }

        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK && data != null) {
            Bundle extras = data.getExtras();
            if (extras != null) {
                bitmap = (Bitmap) extras.get("data");
                if (bitmap != null) {
                    imageView.setImageBitmap(bitmap);
                } else {
                    // Handle jika gambar null
                    Toast.makeText(this, "Gagal mengambil gambar", Toast.LENGTH_SHORT).show();
                }
            } else {
                // Handle jika bundle null
                Toast.makeText(this, "Gagal mengambil gambar", Toast.LENGTH_SHORT).show();
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    private void checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            // Jika izin belum diberikan, minta izin
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_CODE);
        } else {
            // Jika izin sudah diberikan, lanjut ke aksi berikutnya
            dispatchTakePictureIntent();
        }
    }


    private void dispatchTakePictureIntent() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
        } else {
            // Handle jika tidak ada aplikasi kamera yang terinstal
            // Misalnya, Anda dapat menampilkan pesan kesalahan atau menawarkan alternatif lain.
            // Contoh: Menampilkan Toast
            Toast.makeText(this, "Tidak ada aplikasi kamera yang terinstal", Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // Izin diberikan, lanjut ke aksi berikutnya
                dispatchTakePictureIntent();
            } else {
                // Izin ditolak, beri informasi kepada pengguna atau lakukan tindakan lainnya
                Toast.makeText(this, "Izin kamera ditolak", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private float[] bitmapToFloatArray(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        float[] floatValues = new float[width * height * 3];
        int pixel = 0;
        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < height; ++j) {
                final int val = pixels[pixel++];
                floatValues[i * height * 3 + j * 3 + 0] = Color.red(val) / 255.0f;
                floatValues[i * height * 3 + j * 3 + 1] = Color.green(val) / 255.0f;
                floatValues[i * height * 3 + j * 3 + 2] = Color.blue(val) / 255.0f;
            }
        }

        return floatValues;
    }

    // Update runInference method to use the new method
    private void runInference() {
        try {
            // Resize gambar ke ukuran yang diharapkan oleh model
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);

            // Convert Bitmap to float array
            float[] floatValues = bitmapToFloatArray(resizedBitmap);

            // Persiapkan input buffer
            ByteBuffer inputBuffer = ByteBuffer.allocateDirect(floatValues.length * 4);
            inputBuffer.order(ByteOrder.nativeOrder());
            inputBuffer.rewind();
            inputBuffer.asFloatBuffer().put(floatValues);

            // Persiapkan output buffer
            TensorBuffer outputBuffer = TensorBuffer.createFixedSize(new int[]{1, NUM_CLASSES}, DataType.FLOAT32);

            // Jalankan model inference
            tflite.run(inputBuffer, outputBuffer.getBuffer());

            // Tampilkan hasil
            displayResult(outputBuffer.getFloatArray());
        } catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(this, "Gagal menjalankan inference", Toast.LENGTH_SHORT).show();
        }
    }
    private void displayResult(float[] results) {
        // Temukan indeks kelas dengan persentase tertinggi
        float maxPercentage = -1;
        int maxIndex = -1;

        for (int i = 0; i < results.length; i++) {
            if (results[i] > maxPercentage) {
                maxPercentage = results[i];
                maxIndex = i;
            }
        }

        // Tampilkan label dari kelas dengan persentase tertinggi
        String classifiedLabel = maxIndex != -1 ? LABELS[maxIndex] : "Unknown";
        ClassifiedText.setText("Classified As: " + classifiedLabel);
    }


    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (tflite != null) {
            tflite.close();
        }
    }
}