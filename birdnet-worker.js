/* BirdNET in-browser inference worker
 * Adapted from https://github.com/georg95/birdnet-web (MIT-friendly research demo)
 * for Flatwater Collective's bird sound identifier.
 * All inference runs locally in the browser. No audio leaves the device.
 */

importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js')

// Locally-served BirdNET model (in client/public/birdnet/)
const MODEL_URL  = './birdnet/model.json'
const LABELS_URL = './birdnet/labels.txt'

main().catch(e => postMessage({ message: 'fatal_error', error: String(e && e.stack || e) }))

async function main() {
  try {
    await tf.setBackend('webgl')
  } catch (e) {
    await tf.setBackend('cpu')
  }
  await tf.ready()

  const BirdNetJS = await predictModel()
  postMessage({ message: 'warmup', progress: 80 })
  await BirdNetJS.warmup()

  postMessage({ message: 'load_labels', progress: 95 })
  const labelsText = await fetch(LABELS_URL).then(r => r.text())
  const birdsList = labelsText.split('\n').filter(Boolean)
  const birds = new Array(birdsList.length)
  for (let i = 0; i < birdsList.length; i++) {
    const parts = birdsList[i].split('_')
    birds[i] = {
      sci: parts[0] || '',
      name: parts[1] || birdsList[i],
    }
  }

  postMessage({ message: 'loaded' })

  const MIN_AUDIO_CONFIDENCE = 0.06

  onmessage = async function ({ data }) {
    try {
      if (data.message === 'predict') {
        // Boost quiet phone mics: normalize each 3s window before inference.
        const pcm = data.pcmAudio instanceof Float32Array
          ? data.pcmAudio
          : new Float32Array(data.pcmAudio)
        const batches = pcm.length / 144000
        const normalized = new Float32Array(pcm.length)
        for (let batch = 0; batch < batches; batch++) {
          const start = batch * 144000
          let peak = 0
          for (let i = 0; i < 144000; i++) {
            const a = Math.abs(pcm[start + i])
            if (a > peak) peak = a
          }
          const scale = peak > 1e-5 ? 0.92 / peak : 1
          for (let i = 0; i < 144000; i++) {
            normalized[start + i] = pcm[start + i] * scale
          }
        }
        const input = tf.tensor(normalized, [batches, 144000])
        const predictionList = await BirdNetJS.predict(input)
        const prediction = []
        for (let batch = 0; batch < predictionList.length; batch++) {
          for (let i = 0; i < predictionList[batch].length; i++) {
            const confidence = predictionList[batch][i]
            if (confidence > MIN_AUDIO_CONFIDENCE) {
              prediction.push({
                name: birds[i].name,
                sci: birds[i].sci,
                batch,
                confidence,
              })
            }
          }
        }
        // Keep UI/worker chatter light on mobile: rank by confidence and cap hits per clip.
        prediction.sort((a, b) => b.confidence - a.confidence)
        const TOP_PER_PREDICT = 32
        const trimmed = prediction.slice(0, TOP_PER_PREDICT)
        postMessage({ message: 'predict', prediction: trimmed, requestId: data.requestId })
      }
    } catch (err) {
      postMessage({ message: 'error', error: String(err && err.stack || err) })
    }
  }
}

async function predictModel() {
  const BirdNetJS = await tf.loadLayersModel(MODEL_URL, {
    onProgress: progress => postMessage({ message: 'load_model', progress: (progress * 70) | 0 })
  })
  async function predict(signal) {
    const resTensor = BirdNetJS.predict(signal)
    signal.dispose()
    const result = await resTensor.array()
    resTensor.dispose()
    return result
  }
  return {
    async warmup() { await predict(tf.zeros([1, 144000])) },
    predict,
  }
}

/* Custom mel-spectrogram layer required by the BirdNET TFJS model */
class MelSpecLayerSimple extends tf.layers.Layer {
  constructor(config) {
    super(config)
    this.sampleRate = config.sampleRate
    this.specShape = config.specShape
    this.frameStep = config.frameStep
    this.frameLength = config.frameLength
    this.melFilterbank = tf.tensor2d(config.melFilterbank)
  }
  build(inputShape) {
    this.magScale = this.addWeight(
      'magnitude_scaling', [], 'float32',
      tf.initializers.constant({ value: 1.23 })
    )
    super.build(inputShape)
  }
  computeOutputShape(inputShape) {
    return [inputShape[0], this.specShape[0], this.specShape[1], 1]
  }
  call(inputs) {
    return tf.tidy(() => {
      inputs = inputs[0]
      return tf.stack(inputs.split(inputs.shape[0]).map((input) => {
        let spec = input.squeeze()
        spec = tf.sub(spec, tf.min(spec, -1, true))
        spec = tf.div(spec, tf.max(spec, -1, true).add(0.000001))
        spec = tf.sub(spec, 0.5)
        spec = tf.mul(spec, 2.0)
        spec = tf.engine().runKernel('STFT', { signal: spec, frameLength: this.frameLength, frameStep: this.frameStep })
        spec = tf.matMul(spec, this.melFilterbank)
        spec = spec.pow(2.0)
        spec = spec.pow(tf.div(1.0, tf.add(1.0, tf.exp(this.magScale.read()))))
        spec = tf.reverse(spec, -1)
        spec = tf.transpose(spec)
        spec = spec.expandDims(-1)
        return spec
      }))
    })
  }
  static get className() { return 'MelSpecLayerSimple' }
}
tf.serialization.registerClass(MelSpecLayerSimple)

/* WebGL STFT kernel */
tf.registerKernel({
  kernelName: 'STFT',
  backendName: 'webgl',
  kernelFunc: ({ backend, inputs: { signal, frameLength, frameStep } }) => {
    const innerDim = frameLength / 2
    const batch = (signal.size - frameLength + frameStep) / frameStep | 0
    let currentTensor = backend.runWebGLProgram({
      variableNames: ['x'],
      outputShape: [batch, frameLength],
      userCode: `
      void main() {
        ivec2 coords = getOutputCoords();
        int p = coords[1] % ${innerDim};
        int k = 0;
        for (int i = 0; i < ${Math.log2(innerDim)}; ++i) {
          if ((p & (1 << i)) != 0) { k |= (1 << (${Math.log2(innerDim) - 1} - i)); }
        }
        int i = 2 * k;
        if (coords[1] >= ${innerDim}) { i = 2 * (k % ${innerDim}) + 1; }
        int q = coords[0] * ${frameLength} + i;
        float val = getX((q / ${frameLength}) * ${frameStep} + q % ${frameLength});
        float cosArg = ${2.0 * Math.PI / frameLength} * float(q);
        float mul = 0.5 - 0.5 * cos(cosArg);
        setOutput(val * mul);
      }`
    }, [signal], 'float32')
    for (let len = 1; len < innerDim; len *= 2) {
      let prevTensor = currentTensor
      currentTensor = backend.runWebGLProgram({
        variableNames: ['x'],
        outputShape: [batch, innerDim * 2],
        userCode: `void main() {
          ivec2 coords = getOutputCoords();
          int batch = coords[0];
          int i = coords[1];
          int k = i % ${innerDim};
          int isHigh = (k % ${len * 2}) / ${len};
          int highSign = (1 - isHigh * 2);
          int baseIndex = k - isHigh * ${len};
          float t = ${Math.PI / len} * float(k % ${len});
          float a = cos(t);
          float b = sin(-t);
          float oddK_re = getX(batch, baseIndex + ${len});
          float oddK_im = getX(batch, baseIndex + ${len + innerDim});
          if (i < ${innerDim}) {
            float evenK_re = getX(batch, baseIndex);
            setOutput(evenK_re + (oddK_re * a - oddK_im * b) * float(highSign));
          } else {
            float evenK_im = getX(batch, baseIndex + ${innerDim});
            setOutput(evenK_im + (oddK_re * b + oddK_im * a) * float(highSign));
          }
        }`
      }, [currentTensor], 'float32')
      backend.disposeIntermediateTensorInfo(prevTensor)
    }
    const real = backend.runWebGLProgram({
      variableNames: ['x'],
      outputShape: [batch, innerDim + 1],
      userCode: `void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int i = coords[1];
        int zI = i % ${innerDim};
        int conjI = (${innerDim} - i) % ${innerDim};
        float Zk0 = getX(batch, zI);
        float Zk1 = getX(batch, zI+${innerDim});
        float Zk_conj0 = getX(batch, conjI);
        float Zk_conj1 = -getX(batch, conjI+${innerDim});
        float t = ${-2 * Math.PI} * float(i) / float(${innerDim * 2});
        float diff0 = Zk0 - Zk_conj0;
        float diff1 = Zk1 - Zk_conj1;
        float result = (Zk0 + Zk_conj0 + cos(t) * diff1 + sin(t) * diff0) * 0.5;
        setOutput(result);
      }`
    }, [currentTensor], 'float32')
    backend.disposeIntermediateTensorInfo(currentTensor)
    return real
  }
})
