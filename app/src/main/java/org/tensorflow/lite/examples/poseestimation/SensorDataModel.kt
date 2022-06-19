package org.tensorflow.lite.examples.poseestimation

import com.google.gson.annotations.SerializedName

data class SensorDTO(
    @SerializedName("terminalId")
    val terminalId : Long,
    @SerializedName("temper_humid_sensor")
    val temper_humid_sensor : Int,
    @SerializedName("smoke_sensor")
    val smoke_sensor : Int,
    @SerializedName("camera_sensor")
    var camera_sensor : Int = 0,
    @SerializedName("motion_sensor")
    val motion_sensor : Int,
    @SerializedName("illuminance_sensor")
    val illuminance_sensor : Int,
    @SerializedName("flame_sensor")
    val flame_sensor : Int,
    @SerializedName("sound_sensor")
    val sound_sensor : Int,
)

data class PostResult(
    var result:String? = null
)