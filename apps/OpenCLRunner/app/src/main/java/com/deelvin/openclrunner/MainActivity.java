package com.deelvin.openclrunner;

import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetManager;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {

    static {
       System.loadLibrary("openclrunner");
    }

    private AssetManager mgr;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mgr = getResources().getAssets();

        final Button runButton = findViewById(R.id.runButton);
        runButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                runButton.setText("Running...");
                runButton.setEnabled(false);

                final Thread t = new Thread(new Runnable() {
                    @Override
                    public void run() {
                        final TextView timeText = findViewById(R.id.timeView);
                        String msg = runOpenCL(mgr);
                        timeText.post(new Runnable() {
                            public void run() {
                                timeText.setText(msg);
                            }
                        });
                        final Button runButton = findViewById(R.id.runButton);
                        runButton.post(new Runnable() {
                            public void run() {
                                runButton.setText("RUN");
                                runButton.setEnabled(true);
                            }
                        });
                    }
                });
                t.start();
            }
        });
    }

    public native String runOpenCL(AssetManager mgr);
}