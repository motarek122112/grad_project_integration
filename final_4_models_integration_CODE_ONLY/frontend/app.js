// ===============================
// Helper Functions
// ===============================

function setBox(id, data) {
  const box = document.getElementById(id)
  if (!box) return

  if (typeof data === 'string') {
    box.textContent = data
  } else {
    box.textContent = JSON.stringify(data, null, 2)
  }
}

function setText(id, text) {
  const box = document.getElementById(id)
  if (box) box.textContent = text || '---'
}

function setAudio(id, url) {
  const audio = document.getElementById(id)
  if (!audio) return

  if (url) {
    audio.src = url
    audio.style.display = 'block'
    audio.load()
  } else {
    audio.pause()
    audio.removeAttribute('src')
    audio.style.display = 'none'
  }
}

function getSentenceFromResult(result) {
  if (!result) return ''

  return (
    result.lm_sentence ||
    result.full_sentence ||
    result.text ||
    result.clean_text ||
    result.predicted_text ||
    result.transcript ||
    result.word ||
    result.label ||
    (
      result.word_sequence
        ? result.word_sequence.join(' ')
        : ''
    ) ||
    ''
  )
}

function handleSignResult(data, mode) {
  if (!data || !data.ok || !data.result) return

  const result = data.result

  if (mode === 'single') {
    const text =
      result.text ||
      result.clean_text ||
      result.word ||
      result.gloss ||
      ''

    setText('singleSignBox', text)
  }

  if (mode === 'sentence') {
    const sentence = getSentenceFromResult(result)
    setText('fullSentenceBox', sentence)
  }

  if (mode === 'speech') {
    const text =
      result.text ||
      result.lm_sentence ||
      result.full_sentence ||
      result.clean_text ||
      ''

    setText('signSpeechTextBox', text)

    if (result.audio_url) {
      setAudio('signSpeechAudio', result.audio_url)
    }
  }
}


// ===============================
// System Status
// ===============================

async function checkStatus() {
  setBox('statusResult', 'Checking system status...')

  try {
    const response = await fetch('/api/status')
    const data = await response.json()
    setBox('statusResult', data)

  } catch (err) {
    setBox('statusResult', {
      ok: false,
      error: err.message
    })
  }
}


// ===============================
// Video Upload Functions
// ===============================

async function submitVideoFile(endpoint, inputId, resultBoxId, playAudio = false) {
  const input = document.getElementById(inputId)

  if (!input || !input.files.length) {
    alert('Choose a video first')
    return
  }

  setBox(resultBoxId, 'Running... first load may take time.')

  const fd = new FormData()
  fd.append('video', input.files[0])

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      body: fd
    })

    const data = await response.json()
    setBox(resultBoxId, data)

    if (endpoint.includes('/api/sign/single')) {
      handleSignResult(data, 'single')
    }

    if (endpoint.includes('/api/sign/to-speech') || endpoint.includes('/api/sign/tts')) {
      handleSignResult(data, 'speech')

      if (playAudio && data.ok && data.result && data.result.audio_url) {
        const audio = document.getElementById('signSpeechAudio')

        if (audio) {
          audio.src = data.result.audio_url
          audio.style.display = 'block'
          audio.play()
        }
      }
    }

  } catch (err) {
    setBox(resultBoxId, {
      ok: false,
      error: err.message
    })
  }
}


// ===============================
// Multi-Sign Video → Sentence
// ===============================

async function predictSentenceFromFile() {
  const input = document.getElementById('sentenceVideo')

  if (!input || !input.files.length) {
    alert('Choose a video first')
    return
  }

  setText('fullSentenceBox', 'Running...')
  setBox('sentenceResult', 'Running... first load may take time.')

  const fd = new FormData()
  fd.append('video', input.files[0])

  // Settings handled again in backend too.
  fd.append('top_k', '5')
  fd.append('threshold', '0.08')
  fd.append('min_pause_sec', '0.35')
  fd.append('min_segment_sec', '0.30')
  fd.append('confidence_threshold', '0.08')
  fd.append('use_language_decoder', 'true')

  try {
    const response = await fetch('/api/sign/sentence', {
      method: 'POST',
      body: fd
    })

    const data = await response.json()

    setBox('sentenceResult', data)
    handleSignResult(data, 'sentence')

  } catch (err) {
    setText('fullSentenceBox', 'Error')

    setBox('sentenceResult', {
      ok: false,
      error: err.message
    })
  }
}


// ===============================
// Text To Speech
// ===============================

async function textToSpeech() {
  const textInput = document.getElementById('ttsText')
  const text = textInput ? textInput.value.trim() : ''

  if (!text) {
    alert('Write text first')
    return
  }

  setBox('ttsResult', 'Generating speech...')
  setAudio('ttsAudio', null)

  const fd = new FormData()
  fd.append('text', text)

  try {
    const response = await fetch('/api/tts', {
      method: 'POST',
      body: fd
    })

    const data = await response.json()
    setBox('ttsResult', data)

    if (data.ok && data.result && data.result.audio_url) {
      setAudio('ttsAudio', data.result.audio_url)
    }

  } catch (err) {
    setBox('ttsResult', {
      ok: false,
      error: err.message
    })
  }
}


// ===============================
// Speech To Text
// ===============================

async function speechToText() {
  const input = document.getElementById('speechAudio')

  if (!input || !input.files.length) {
    alert('Choose an audio file first')
    return
  }

  setText('speechTextBox', 'Running...')
  setBox('speechResult', 'Transcribing...')

  const fd = new FormData()
  fd.append('audio', input.files[0])

  // Greedy is usually more stable for your current prototype.
  fd.append('decoder', 'greedy')

  try {
    const response = await fetch('/api/speech', {
      method: 'POST',
      body: fd
    })

    const data = await response.json()
    setBox('speechResult', data)

    if (data.ok && data.result) {
      const text =
        data.result.text ||
        data.result.clean_text ||
        data.result.transcript ||
        ''

      setText('speechTextBox', text)
    } else {
      setText('speechTextBox', 'Error')
    }

  } catch (err) {
    setText('speechTextBox', 'Error')

    setBox('speechResult', {
      ok: false,
      error: err.message
    })
  }
}


// ===============================
// Age & Gender
// ===============================

async function ageGenderPredict() {
  const input = document.getElementById('ageImage')

  if (!input || !input.files.length) {
    alert('Choose an image first')
    return
  }

  setText('ageGenderBox', 'Running...')
  setBox('ageResult', 'Predicting age and gender...')

  const fd = new FormData()
  fd.append('image', input.files[0])

  try {
    const response = await fetch('/api/age-gender', {
      method: 'POST',
      body: fd
    })

    const data = await response.json()
    setBox('ageResult', data)

    if (data.ok && data.result) {
      const result = data.result

      const age =
        result.age ||
        result.predicted_age ||
        result.age_prediction ||
        result.age_range ||
        'Unknown'

      const gender =
        result.gender ||
        result.predicted_gender ||
        result.gender_prediction ||
        'Unknown'

      setText('ageGenderBox', 'Age: ' + age + ' | Gender: ' + gender)

    } else {
      setText('ageGenderBox', 'Error')
    }

  } catch (err) {
    setText('ageGenderBox', 'Error')

    setBox('ageResult', {
      ok: false,
      error: err.message
    })
  }
}


// ===============================
// Camera Recorder For Sign Videos
// ===============================

let cameraStream = null
let mediaRecorder = null
let recordedChunks = []
let recordedBlob = null

let activeCameraEndpoint = null
let activeCameraResultBox = null
let activeCameraMode = null


function getBrowserCameraStream(constraints) {
  if (
    navigator.mediaDevices &&
    typeof navigator.mediaDevices.getUserMedia === 'function'
  ) {
    return navigator.mediaDevices.getUserMedia(constraints)
  }

  const oldGetUserMedia =
    navigator.webkitGetUserMedia ||
    navigator.mozGetUserMedia ||
    navigator.msGetUserMedia

  if (typeof oldGetUserMedia === 'function') {
    return new Promise(function(resolve, reject) {
      oldGetUserMedia.call(navigator, constraints, resolve, reject)
    })
  }

  return Promise.reject(
    new Error('Camera API is not supported. Please use Chrome or Edge and open http://127.0.0.1:8000')
  )
}


async function openSignCamera(endpoint, resultBoxId, mode = 'single') {
  activeCameraEndpoint = endpoint
  activeCameraResultBox = resultBoxId
  activeCameraMode = mode

  recordedChunks = []
  recordedBlob = null

  const modal = document.getElementById('cameraModal')
  const status = document.getElementById('recordStatus')
  const sendBtn = document.getElementById('sendRecordBtn')
  const stopBtn = document.getElementById('stopRecordBtn')
  const preview = document.getElementById('cameraPreview')

  if (!modal || !status || !sendBtn || !stopBtn || !preview) {
    alert('Camera UI is missing in index.html')
    return
  }

  modal.classList.remove('hidden')
  status.textContent = 'Opening camera...'
  sendBtn.disabled = true
  stopBtn.disabled = true

  preview.controls = false
  preview.src = ''
  preview.srcObject = null

  try {
    let constraints = {
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30 },
        facingMode: 'user'
      },
      audio: false
    }

    try {
      cameraStream = await getBrowserCameraStream(constraints)
    } catch (highQualityError) {
      constraints = {
        video: true,
        audio: false
      }

      cameraStream = await getBrowserCameraStream(constraints)
    }

    preview.srcObject = cameraStream
    preview.muted = true
    preview.playsInline = true

    await preview.play()

    status.textContent =
      'Camera ready. Use good light and keep both hands inside the frame.'

  } catch (err) {
    console.error('Camera error:', err)

    status.textContent =
      'Camera error: ' + (err.message || err.name || String(err))
  }
}


function startSignRecording() {
  if (!cameraStream) {
    alert('Camera is not ready')
    return
  }

  if (typeof MediaRecorder === 'undefined') {
    alert('MediaRecorder is not supported in this browser. Use Chrome or Edge.')
    return
  }

  recordedChunks = []
  recordedBlob = null

  let options = {}

  if (MediaRecorder.isTypeSupported('video/webm;codecs=vp8')) {
    options = { mimeType: 'video/webm;codecs=vp8' }
  } else if (MediaRecorder.isTypeSupported('video/webm;codecs=vp9')) {
    options = { mimeType: 'video/webm;codecs=vp9' }
  } else if (MediaRecorder.isTypeSupported('video/webm')) {
    options = { mimeType: 'video/webm' }
  }

  try {
    mediaRecorder = new MediaRecorder(cameraStream, options)
  } catch (err) {
    alert('Could not start recorder: ' + err.message)
    return
  }

  mediaRecorder.ondataavailable = function(event) {
    if (event.data && event.data.size > 0) {
      recordedChunks.push(event.data)
    }
  }

  mediaRecorder.onstop = function() {
    const mimeType = mediaRecorder.mimeType || 'video/webm'
    recordedBlob = new Blob(recordedChunks, { type: mimeType })

    const preview = document.getElementById('cameraPreview')

    preview.srcObject = null
    preview.src = URL.createObjectURL(recordedBlob)
    preview.muted = false
    preview.controls = true
    preview.play()

    document.getElementById('sendRecordBtn').disabled = false
    document.getElementById('recordStatus').textContent =
      'Recording finished. Click Use This Video.'
  }

  mediaRecorder.start(100)

  document.getElementById('stopRecordBtn').disabled = false
  document.getElementById('sendRecordBtn').disabled = true
  document.getElementById('recordStatus').textContent =
    'Recording... make the sign slowly and clearly.'
}


function stopSignRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop()
  }

  const stopBtn = document.getElementById('stopRecordBtn')
  if (stopBtn) stopBtn.disabled = true
}


async function sendRecordedSign() {
  if (!recordedBlob) {
    alert('Record a video first')
    return
  }

  setBox(activeCameraResultBox, 'Uploading recorded video...')

  const status = document.getElementById('recordStatus')
  if (status) {
    status.textContent = 'Sending video to model...'
  }

  let extension = 'webm'

  if (recordedBlob.type.includes('mp4')) {
    extension = 'mp4'
  }

  const recordedFile = new File(
    [recordedBlob],
    'recorded_sign_' + Date.now() + '.' + extension,
    { type: recordedBlob.type }
  )

  const fd = new FormData()
  fd.append('video', recordedFile)

  if (activeCameraMode === 'sentence') {
    fd.append('top_k', '5')
    fd.append('threshold', '0.08')
    fd.append('min_pause_sec', '0.35')
    fd.append('min_segment_sec', '0.30')
    fd.append('confidence_threshold', '0.08')
    fd.append('use_language_decoder', 'true')
  }

  if (activeCameraMode === 'speech') {
    fd.append('sentence_mode', 'false')
    fd.append('use_language_decoder', 'true')
  }

  try {
    const response = await fetch(activeCameraEndpoint, {
      method: 'POST',
      body: fd
    })

    const data = await response.json()

    setBox(activeCameraResultBox, data)
    handleSignResult(data, activeCameraMode)

    if (activeCameraMode === 'speech' && data.ok && data.result && data.result.audio_url) {
      const audio = document.getElementById('signSpeechAudio')

      if (audio) {
        audio.src = data.result.audio_url
        audio.style.display = 'block'
        audio.play()
      }
    }

    if (status) {
      status.textContent = 'Done. Result is shown below.'
    }

  } catch (err) {
    setBox(activeCameraResultBox, {
      ok: false,
      error: err.message
    })

    if (status) {
      status.textContent = 'Error: ' + err.message
    }
  }
}


function closeSignCamera() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop()
  }

  if (cameraStream) {
    cameraStream.getTracks().forEach(function(track) {
      track.stop()
    })
  }

  cameraStream = null
  mediaRecorder = null
  recordedChunks = []
  recordedBlob = null

  const preview = document.getElementById('cameraPreview')

  if (preview) {
    preview.srcObject = null
    preview.src = ''
    preview.controls = false
  }

  const modal = document.getElementById('cameraModal')

  if (modal) {
    modal.classList.add('hidden')
  }
}