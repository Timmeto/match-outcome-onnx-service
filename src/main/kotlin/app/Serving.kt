package app
import ai.onnxruntime.*
import org.springframework.web.bind.annotation.*

data class PredictRequest(val features: List<Double>)
data class PredictResponse(val winProb: Double)

@RestController
@RequestMapping("/api")
class ModelController {
    private val env = OrtEnvironment.getEnvironment()
    private val session = env.createSession("models/match_baseline.onnx", OrtSession.SessionOptions())

    @PostMapping("/winprob")
    fun predict(@RequestBody req: PredictRequest): PredictResponse {
        val input = OnnxTensor.createTensor(env, arrayOf(req.features.toDoubleArray()))
        session.use { s ->
            val out = s.run(mapOf("input" to input))
            val prob = (out[0].value as Array<FloatArray>)[0][0].toDouble()
            return PredictResponse(prob)
        }
    }
}
