/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================
*/

package org.tensorflow.lite.examples.poseestimation

import android.Manifest
import android.app.AlertDialog
import android.app.Dialog
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.os.Process
import android.view.SurfaceView
import android.view.View
import android.view.WindowManager
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.SwitchCompat
import androidx.core.content.ContextCompat
import androidx.fragment.app.DialogFragment
import androidx.lifecycle.lifecycleScope
import com.github.mikephil.charting.charts.LineChart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.tensorflow.lite.examples.poseestimation.camera.CameraSource
import org.tensorflow.lite.examples.poseestimation.data.Device
import org.tensorflow.lite.examples.poseestimation.ml.*
import com.github.mikephil.charting.components.YAxis

import com.github.mikephil.charting.components.Legend

import android.R.attr.name
import com.github.mikephil.charting.components.Description
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet

import android.R.attr.name
import android.annotation.SuppressLint
import com.github.mikephil.charting.interfaces.datasets.ILineDataSet

import android.R.attr.name
import android.util.Log
import androidx.annotation.StringRes
import com.github.mikephil.charting.data.Entry
import com.google.gson.Gson
import com.harrysoft.androidbluetoothserial.BluetoothManager
import com.harrysoft.androidbluetoothserial.BluetoothSerialDevice
import com.harrysoft.androidbluetoothserial.SimpleBluetoothDeviceInterface
import com.harrysoft.androidbluetoothserial.SimpleBluetoothDeviceInterface.OnMessageReceivedListener
import com.harrysoft.androidbluetoothserial.SimpleBluetoothDeviceInterface.OnMessageSentListener
import io.reactivex.android.schedulers.AndroidSchedulers
import io.reactivex.disposables.CompositeDisposable
import io.reactivex.schedulers.Schedulers
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.lang.StringBuilder
import java.util.Timer
import java.util.TimerTask

class MainActivity : AppCompatActivity() {
    companion object {
        private const val FRAGMENT_DIALOG = "dialog"
    }

    /** A [SurfaceView] for camera preview.   */
    private lateinit var surfaceView: SurfaceView

    /** Default pose estimation model is 1 (MoveNet Thunder)
     * 0 == MoveNet Lightning model
     * 1 == MoveNet Thunder model
     * 2 == MoveNet MultiPose model
     * 3 == PoseNet model
     **/
    private var modelPos = 1

    // variable for custom
    private val sampleFrame = 15
    private val dangerCutoff = 10
    private var sampleBufferValue = 0
    private var frameCurrent = 0
    private lateinit var chart: LineChart
    private var bluetoothManager: BluetoothManager? = null
    private var deviceInterface: SimpleBluetoothDeviceInterface? = null
    private var mac: String = ""
    private val api = APIS.create()
    private var data: SensorDTO? = null
    private lateinit var timer: Timer
    private lateinit var timerTask: TimerTask

    // A CompositeDisposable that keeps track of all of our asynchronous tasks
    private val compositeDisposable: CompositeDisposable = CompositeDisposable()

    /** Default device is CPU */
    private var device = Device.CPU

    private lateinit var tvScore: TextView
    private lateinit var tvFPS: TextView
    private lateinit var spnDevice: Spinner
    private lateinit var spnModel: Spinner
    private lateinit var spnTracker: Spinner
    private lateinit var vTrackerOption: View
    private lateinit var tvClassificationValue1: TextView

    //    private lateinit var tvClassificationValue2: TextView
//    private lateinit var tvClassificationValue3: TextView
    private lateinit var swClassification: SwitchCompat
    private lateinit var vClassificationOption: View
    private var cameraSource: CameraSource? = null
    private var isClassifyPose = false
    private val requestPermissionLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { isGranted: Boolean ->
            if (isGranted) {
                // Permission is granted. Continue the action or workflow in your
                // app.
                openCamera()
            } else {
                // Explain to the user that the feature is unavailable because the
                // features requires a permission that the user has denied. At the
                // same time, respect the user's decision. Don't link to system
                // settings in an effort to convince the user to change their
                // decision.
                ErrorDialog.newInstance(getString(R.string.tfe_pe_request_permission))
                    .show(supportFragmentManager, FRAGMENT_DIALOG)
            }
        }
    private var changeModelListener = object : AdapterView.OnItemSelectedListener {
        override fun onNothingSelected(parent: AdapterView<*>?) {
            // do nothing
        }

        override fun onItemSelected(
            parent: AdapterView<*>?,
            view: View?,
            position: Int,
            id: Long
        ) {
            changeModel(position)
        }
    }

    private var changeDeviceListener = object : AdapterView.OnItemSelectedListener {
        override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
            changeDevice(position)
        }

        override fun onNothingSelected(parent: AdapterView<*>?) {
            // do nothing
        }
    }

    private var changeTrackerListener = object : AdapterView.OnItemSelectedListener {
        override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
            changeTracker(position)
        }

        override fun onNothingSelected(parent: AdapterView<*>?) {
            // do nothing
        }
    }

    private var setClassificationListener =
        CompoundButton.OnCheckedChangeListener { _, isChecked ->
            showClassificationResult(isChecked)
            isClassifyPose = isChecked
            isPoseClassifier()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        // keep screen on while app is running
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        tvScore = findViewById(R.id.tvScore)
        tvFPS = findViewById(R.id.tvFps)
        spnModel = findViewById(R.id.spnModel)
        spnDevice = findViewById(R.id.spnDevice)
        spnTracker = findViewById(R.id.spnTracker)
        vTrackerOption = findViewById(R.id.vTrackerOption)
        surfaceView = findViewById(R.id.surfaceView)
        tvClassificationValue1 = findViewById(R.id.tvClassificationValue1)
//        tvClassificationValue2 = findViewById(R.id.tvClassificationValue2)
//        tvClassificationValue3 = findViewById(R.id.tvClassificationValue3)
        swClassification = findViewById(R.id.swPoseClassification)
        vClassificationOption = findViewById(R.id.vClassificationOption)
        initSpinner()
        spnModel.setSelection(modelPos)
        swClassification.setOnCheckedChangeListener(setClassificationListener)
        if (!isCameraPermissionGranted()) {
            requestPermission()
        }

        // custom code
        chart = findViewById(R.id.LineChart);
        chartInit()

        // Setup our BluetoothManager
        bluetoothManager = BluetoothManager.instance
        if (bluetoothManager == null) {
            // Bluetooth unavailable on this device :( tell the user
            Toast.makeText(application, R.string.no_bluetooth, Toast.LENGTH_LONG).show()
        } else {
            bluetoothManager?.pairedDevices?.forEach {
                Log.d("PairedDevices:", it.name + ": " + it.address)
//                if(it.equals("센서 단말기")){
                mac = it.address
//                }
            }

            Log.d("PairedDevices:", mac)
            // Connect asynchronously
            compositeDisposable.add(
                bluetoothManager!!.openSerialDevice(mac)
                    .subscribeOn(Schedulers.io())
                    .observeOn(AndroidSchedulers.mainThread())
                    .subscribe(
                        { device: BluetoothSerialDevice ->
                            onConnected(
                                device.toSimpleDeviceInterface()
                            )
                        }
                    ) { t: Throwable? ->
                        toast("여는 것에 실패!")
                    })
        }

    }


    override fun onStart() {
        super.onStart()
        openCamera()
    }

    override fun onResume() {
        cameraSource?.resume()
        super.onResume()
    }

    override fun onPause() {
        cameraSource?.close()
        cameraSource = null
        super.onPause()
    }

    private fun chartInit() {
        chart.setDrawGridBackground(true)
        chart.setBackgroundColor(Color.BLACK)
        chart.setGridBackgroundColor(Color.BLACK)


        // description text
        chart.description.isEnabled = true
        val des: Description = chart.description
        des.setEnabled(true)
        des.setText("Real-Time DATA")
        des.setTextSize(15f)
        des.setTextColor(Color.WHITE)

        // touch gestures (false-비활성화)
        chart.setTouchEnabled(false)

        // scaling and dragging (false-비활성화)
        chart.isDragEnabled = false
        chart.setScaleEnabled(false)

        //auto scale
        chart.isAutoScaleMinMaxEnabled = true

        // if disabled, scaling can be done on x- and y-axis separately
        chart.setPinchZoom(false)

        //X축
        chart.xAxis.setDrawGridLines(true)
        chart.xAxis.setDrawAxisLine(false)

        chart.xAxis.isEnabled = true
        chart.xAxis.setDrawGridLines(false)

        //Legend
        val l = chart.legend
        l.isEnabled = true
        l.formSize = 10f // set the size of the legend forms/shapes

        l.textSize = 12f
        l.textColor = Color.WHITE

        //Y축
        val leftAxis = chart.axisLeft
        leftAxis.isEnabled = true
//        leftAxis.textColor = resources.getColor(R.color.grid)
        leftAxis.textColor = Color.WHITE
        leftAxis.setDrawGridLines(true)
//        leftAxis.gridColor = resources.getColor(R.color.colorGrid)
        leftAxis.gridColor = Color.WHITE

        val rightAxis = chart.axisRight
        rightAxis.isEnabled = false

        // don't forget to refresh the drawing
        chart.invalidate()

    }

    private fun addEntry(num: Int) {
        var data = chart.data
        if (data == null) {
            data = LineData()
            chart.data = data
        }
        var set = data.getDataSetByIndex(0)
        // set.addEntry(...); // can be called as well
        if (set == null) {
            set = createSet()
            data.addDataSet(set)
        }

        data.addEntry(Entry(set!!.entryCount.toFloat(), num.toFloat()), 0)
        data.notifyDataChanged()


        // let the chart know it's data has changed
        chart.notifyDataSetChanged()
        chart.setVisibleXRangeMaximum(sampleFrame.toFloat() + 5)
        // this automatically refreshes the chart (calls invalidate())
        chart.moveViewTo(data.entryCount.toFloat(), 50f, YAxis.AxisDependency.LEFT)

        if(num>=dangerCutoff && timer != null){
            timerTask.run()
        }
    }


    @SuppressLint("ResourceType")
    private fun createSet(): LineDataSet? {
        val set = LineDataSet(null, "Real-time Line Data")
        set.lineWidth = 1f
        set.setDrawValues(false)
//        set.valueTextColor = resources.getColor(Color.WHITE)
//        set.color = resources.getColor(Color.WHITE)
        set.valueTextColor = Color.WHITE
        set.color = Color.WHITE

        set.mode = LineDataSet.Mode.LINEAR
        set.setDrawCircles(false)
        set.highLightColor = Color.rgb(190, 190, 190)
        return set
    }

    private fun onConnected(deviceInterface: SimpleBluetoothDeviceInterface) {
        this.deviceInterface = deviceInterface

        if (deviceInterface != null) {
            // Setup the listeners for the interface
            this.deviceInterface!!.setListeners(
                object : OnMessageReceivedListener {
                    override fun onMessageReceived(message: String) {
                        Log.d("receivedMessage: ", message + data?.toString())
                        val gson = Gson()
                        try {
                            data = gson.fromJson(message, SensorDTO::class.java)

                        } catch(e: Exception){
                            Log.d("receivedMessage: ", e.message!!)
                        }

                    }
                },
                object : OnMessageSentListener {
                    override fun onMessageSent(message: String) {
                        toast("Sent message: " + message)
                    }
                },
                object : SimpleBluetoothDeviceInterface.OnErrorListener {
                    override fun onError(error: Throwable) {
                        toast(R.string.connection_failed)
                    }
                }
            )

            // set Interval
            Log.d("setInterval", "OK")
            try {
                timer = Timer()
                timerTask = newTimerTask()
                timer.schedule(timerTask, 0, 10000)
            } catch (e: Exception) {
                Log.d("timer Error", e.message!!)
            }

            // Tell the user we are connected.
            // Reset the conversation
        } else {
            toast(R.string.connection_failed)
            // deviceInterface was null, so the connection failed
        }
    }

    private fun newTimerTask(): TimerTask {
        return object: TimerTask() {
            override fun run() {
                sendData()
            }
        }
    }

    private fun sendData() {
        Log.d("postResult: ", chart.data?.toString() + " and " + data?.toString() )

        if (chart.data != null && data != null) {

            var chart_dataset = chart.data.getDataSetByIndex(0)
            val chart_last_data = chart_dataset.getEntryForIndex(chart_dataset.entryCount - 1)
            data!!.camera_sensor = chart_last_data.y.toInt()
            Log.d("dataCheck: ", data.toString())
            api.post_sensor_data(data!!).enqueue(object : Callback<PostResult> {
                override fun onResponse(
                    call: Call<PostResult>,
                    response: Response<PostResult>
                ) {
                    Log.d("postResult: ", response.toString())
                    Log.d("postResult: ", response.body().toString())
                }

                override fun onFailure(call: Call<PostResult>, t: Throwable) {
                    Log.d("postResult_Fail: ", t.message.toString())
                }
            })
        }
    }

    private fun toast(@StringRes messageResource: Int) {
        Toast.makeText(application, messageResource, Toast.LENGTH_LONG).show()
    }

    private fun toast(messageResource: String) {
        Toast.makeText(application, messageResource, Toast.LENGTH_LONG).show()
    }

    private fun onMessageReceived(message: String?) {
        toast("sended message: " + message)
    }

    // Adds a sent message to the conversation
    private fun onMessageSent(message: String?) {
        // Add it to the conversation
        // Reset the message box
    }


    // check if permission is granted or not.
    private fun isCameraPermissionGranted(): Boolean {
        return checkPermission(
            Manifest.permission.CAMERA,
            Process.myPid(),
            Process.myUid()
        ) == PackageManager.PERMISSION_GRANTED
    }

    // open camera
    private fun openCamera() {
        if (isCameraPermissionGranted()) {
            if (cameraSource == null) {
                cameraSource =
                    CameraSource(surfaceView, object : CameraSource.CameraSourceListener {
                        override fun onFPSListener(fps: Int) {
                            tvFPS.text = getString(R.string.tfe_pe_tv_fps, fps)
                        }

                        override fun onDetectedInfo(
                            personScore: Float?,
                            poseLabels: List<Pair<String, Float>>?
                        ) {
                            tvScore.text = getString(R.string.tfe_pe_tv_score, personScore ?: 0f)
                            if (personScore ?: 0f < 0.4f) {
                                return
                            }
                            poseLabels?.sortedByDescending { it.second }?.let {
                                if (frameCurrent++ < sampleFrame) {
                                    sampleBufferValue += convertPoseLabels(
                                        if (it.isNotEmpty()) it[0] else null,
                                        personScore ?: 0f
                                    )
                                } else {
                                    tvClassificationValue1.text = getString(
                                        R.string.tfe_pe_tv_classification_value,
                                        sampleBufferValue
                                    )
                                    addEntry(sampleBufferValue)
                                    sampleBufferValue = 0
                                    frameCurrent = 0
                                }

//                                tvClassificationValue2.text = getString(
//                                    R.string.tfe_pe_tv_classification_value,
//                                    convertPoseLabels(if (it.size >= 2) it[1] else null)
//                                )
//                                tvClassificationValue3.text = getString(
//                                    R.string.tfe_pe_tv_classification_value,
//                                    convertPoseLabels(if (it.size >= 3) it[2] else null)
//                                )
                            }
                        }

                    }).apply {
                        prepareCamera()
                    }
                isPoseClassifier()
                lifecycleScope.launch(Dispatchers.Main) {
                    cameraSource?.initCamera()
                }
            }
            createPoseEstimator()
        }
    }

    private fun convertPoseLabels(pair: Pair<String, Float>?, personScore: Float): Int {
        if (pair == null || !pair.first.equals("cobra")) return 0
//        return "${pair.first} (${String.format("%.2f", pair.second)})"
        if (pair.second <= 0.5f) {
            return 0
        } else {
            return 1
        }
    }

    private fun isPoseClassifier() {
        cameraSource?.setClassifier(if (isClassifyPose) PoseClassifier.create(this) else null)
    }

    // Initialize spinners to let user select model/accelerator/tracker.
    private fun initSpinner() {
        ArrayAdapter.createFromResource(
            this,
            R.array.tfe_pe_models_array,
            android.R.layout.simple_spinner_item
        ).also { adapter ->
            // Specify the layout to use when the list of choices appears
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            // Apply the adapter to the spinner
            spnModel.adapter = adapter
            spnModel.onItemSelectedListener = changeModelListener
        }

        ArrayAdapter.createFromResource(
            this,
            R.array.tfe_pe_device_name, android.R.layout.simple_spinner_item
        ).also { adaper ->
            adaper.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)

            spnDevice.adapter = adaper
            spnDevice.onItemSelectedListener = changeDeviceListener
        }

        ArrayAdapter.createFromResource(
            this,
            R.array.tfe_pe_tracker_array, android.R.layout.simple_spinner_item
        ).also { adaper ->
            adaper.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)

            spnTracker.adapter = adaper
            spnTracker.onItemSelectedListener = changeTrackerListener
        }
    }

    // Change model when app is running
    private fun changeModel(position: Int) {
        if (modelPos == position) return
        modelPos = position
        createPoseEstimator()
    }

    // Change device (accelerator) type when app is running
    private fun changeDevice(position: Int) {
        val targetDevice = when (position) {
            0 -> Device.CPU
            1 -> Device.GPU
            else -> Device.NNAPI
        }
        if (device == targetDevice) return
        device = targetDevice
        createPoseEstimator()
    }

    // Change tracker for Movenet MultiPose model
    private fun changeTracker(position: Int) {
        cameraSource?.setTracker(
            when (position) {
                1 -> TrackerType.BOUNDING_BOX
                2 -> TrackerType.KEYPOINTS
                else -> TrackerType.OFF
            }
        )
    }

    private fun createPoseEstimator() {
        // For MoveNet MultiPose, hide score and disable pose classifier as the model returns
        // multiple Person instances.
        val poseDetector = when (modelPos) {
            0 -> {
                // MoveNet Lightning (SinglePose)
                showPoseClassifier(true)
                showDetectionScore(true)
                showTracker(false)
                MoveNet.create(this, device, ModelType.Lightning)
            }
            1 -> {
                // MoveNet Thunder (SinglePose)
                showPoseClassifier(true)
                showDetectionScore(true)
                showTracker(false)
                MoveNet.create(this, device, ModelType.Thunder)
            }
            2 -> {
                // MoveNet (Lightning) MultiPose
                showPoseClassifier(false)
                showDetectionScore(false)
                // Movenet MultiPose Dynamic does not support GPUDelegate
                if (device == Device.GPU) {
                    showToast(getString(R.string.tfe_pe_gpu_error))
                }
                showTracker(true)
                MoveNetMultiPose.create(
                    this,
                    device,
                    Type.Dynamic
                )
            }
            3 -> {
                // PoseNet (SinglePose)
                showPoseClassifier(true)
                showDetectionScore(true)
                showTracker(false)
                PoseNet.create(this, device)
            }
            else -> {
                null
            }
        }
        poseDetector?.let { detector ->
            cameraSource?.setDetector(detector)
        }
    }

    // Show/hide the pose classification option.
    private fun showPoseClassifier(isVisible: Boolean) {
        vClassificationOption.visibility = if (isVisible) View.VISIBLE else View.GONE
        if (!isVisible) {
            swClassification.isChecked = false
        }
    }

    // Show/hide the detection score.
    private fun showDetectionScore(isVisible: Boolean) {
        tvScore.visibility = if (isVisible) View.VISIBLE else View.GONE
    }

    // Show/hide classification result.
    private fun showClassificationResult(isVisible: Boolean) {
        val visibility = if (isVisible) View.VISIBLE else View.GONE
        tvClassificationValue1.visibility = visibility
//        tvClassificationValue2.visibility = visibility
//        tvClassificationValue3.visibility = visibility
    }

    // Show/hide the tracking options.
    private fun showTracker(isVisible: Boolean) {
        if (isVisible) {
            // Show tracker options and enable Bounding Box tracker.
            vTrackerOption.visibility = View.VISIBLE
            spnTracker.setSelection(1)
        } else {
            // Set tracker type to off and hide tracker option.
            vTrackerOption.visibility = View.GONE
            spnTracker.setSelection(0)
        }
    }

    private fun requestPermission() {
        when (PackageManager.PERMISSION_GRANTED) {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) -> {
                // You can use the API that requires the permission.
                openCamera()
            }
            else -> {
                // You can directly ask for the permission.
                // The registered ActivityResultCallback gets the result of this request.
                requestPermissionLauncher.launch(
                    Manifest.permission.CAMERA
                )
            }
        }
    }

    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
    }

    /**
     * Shows an error message dialog.
     */
    class ErrorDialog : DialogFragment() {

        override fun onCreateDialog(savedInstanceState: Bundle?): Dialog =
            AlertDialog.Builder(activity)
                .setMessage(requireArguments().getString(ARG_MESSAGE))
                .setPositiveButton(android.R.string.ok) { _, _ ->
                    // do nothing
                }
                .create()

        companion object {

            @JvmStatic
            private val ARG_MESSAGE = "message"

            @JvmStatic
            fun newInstance(message: String): ErrorDialog = ErrorDialog().apply {
                arguments = Bundle().apply { putString(ARG_MESSAGE, message) }
            }
        }
    }
}
