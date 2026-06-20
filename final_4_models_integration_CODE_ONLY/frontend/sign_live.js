// ======================================================
// Live Near Real-Time Sign Translation - Faster Smooth Version
// Uses existing backend endpoint: /api/sign/single
// ======================================================


// ===============================
// Live Settings
// ===============================

const LIVE_ENDPOINT = '/api/sign-v2/single';

// Faster and smoother than 1700ms
const LIVE_CHUNK_MS = 1000;
const LIVE_LOOP_GAP_MS = 200;
const LIVE_NO_HAND_GAP_MS = 500;
const LIVE_AFTER_ACCEPT_GAP_MS = 600;

// Recognition filtering
const LIVE_MIN_CONFIDENCE = 0.30;

// لو الإيد اختفت بعد توقع مقبول مبدئيًا، نعتمد الكلمة
const LIVE_RELEASE_CONFIRM_CONFIDENCE = 0.15;

// لو الإيد لسه موجودة والحركة مستمرة، لازم ثقة أعلى
const LIVE_HOLD_CONFIRM_CONFIDENCE = 0.62;

// كام مرة نفس التوقع يظهر وهو لسه عامل الحركة
const LIVE_STABLE_COUNT_REQUIRED = 1;

// منع تكرار نفس الكلمة بسرعة
const LIVE_REPEAT_COOLDOWN_MS = 2500;

// هنوقف القبول السريع المباشر
const LIVE_FAST_ACCEPT_CONFIDENCE = 0.99;


// ===============================
// Live State
// ===============================

let liveStream = null;
let liveIsRunning = false;
let liveIsProcessing = false;

let liveGlosses = [];
let liveWords = [];

let liveLastAcceptedGloss = '';
let liveLastAcceptedTime = 0;

let liveCandidateGloss = '';
let liveCandidateCount = 0;
let liveCandidatePred = null;

let liveLoopTimer = null;


// ===============================
// Basic UI Helpers
// ===============================

function liveSetText(id, text) {
  const el = document.getElementById(id);

  if (el) {
    el.textContent = text || '---';
  }
}

function liveSetJSON(id, data) {
  const el = document.getElementById(id);

  if (!el) {
    return;
  }

  if (typeof data === 'string') {
    el.textContent = data;
  } else {
    el.textContent = JSON.stringify(data, null, 2);
  }
}

function liveSetButton(id, disabled) {
  const btn = document.getElementById(id);

  if (btn) {
    btn.disabled = disabled;
  }
}


// ===============================
// Camera Helpers
// ===============================

function liveGetBrowserCameraStream(constraints) {
  if (
    navigator.mediaDevices &&
    typeof navigator.mediaDevices.getUserMedia === 'function'
  ) {
    return navigator.mediaDevices.getUserMedia(constraints);
  }

  const oldGetUserMedia =
    navigator.webkitGetUserMedia ||
    navigator.mozGetUserMedia ||
    navigator.msGetUserMedia;

  if (typeof oldGetUserMedia === 'function') {
    return new Promise(function(resolve, reject) {
      oldGetUserMedia.call(navigator, constraints, resolve, reject);
    });
  }

  return Promise.reject(
    new Error('Camera API is not supported. Please use Chrome or Edge and open http://127.0.0.1:8000')
  );
}

async function openLiveTranslator() {
  const preview = document.getElementById('livePreview');

  if (!preview) {
    alert('livePreview video element not found in index.html');
    return;
  }

  liveSetText('liveStatus', 'Opening camera...');

  try {
    // Lower resolution makes MediaPipe much faster.
    let constraints = {
      video: {
        width: { ideal: 480 },
        height: { ideal: 360 },
        frameRate: { ideal: 12, max: 15 },
        facingMode: 'user'
      },
      audio: false
    };

    try {
      liveStream = await liveGetBrowserCameraStream(constraints);
    } catch (err) {
      constraints = {
        video: true,
        audio: false
      };

      liveStream = await liveGetBrowserCameraStream(constraints);
    }

    preview.srcObject = liveStream;
    preview.muted = true;
    preview.playsInline = true;

    await preview.play();

    liveSetText(
      'liveStatus',
      'Camera ready. Click Start Live and perform one sign clearly every 1–1.5 seconds.'
    );

    liveSetButton('liveStartBtn', false);
    liveSetButton('liveStopBtn', true);

  } catch (err) {
    console.error('Live camera error:', err);

    liveSetText(
      'liveStatus',
      'Camera error: ' + (err.message || err.name || String(err))
    );
  }
}

function closeLiveCamera() {
  stopLiveTranslation();

  if (liveStream) {
    liveStream.getTracks().forEach(function(track) {
      track.stop();
    });
  }

  liveStream = null;

  const preview = document.getElementById('livePreview');

  if (preview) {
    preview.srcObject = null;
  }

  liveSetButton('liveStartBtn', true);
  liveSetButton('liveStopBtn', true);
  liveSetText('liveStatus', 'Camera is closed.');
}


// ===============================
// MediaRecorder Chunk Capture
// ===============================

function liveGetRecorderOptions() {
  if (typeof MediaRecorder === 'undefined') {
    return null;
  }

  if (MediaRecorder.isTypeSupported('video/webm;codecs=vp8')) {
    return { mimeType: 'video/webm;codecs=vp8' };
  }

  if (MediaRecorder.isTypeSupported('video/webm;codecs=vp9')) {
    return { mimeType: 'video/webm;codecs=vp9' };
  }

  if (MediaRecorder.isTypeSupported('video/webm')) {
    return { mimeType: 'video/webm' };
  }

  return {};
}

function recordLiveChunk(stream, durationMs) {
  return new Promise(function(resolve, reject) {
    if (!stream) {
      reject(new Error('Camera stream is not available.'));
      return;
    }

    if (typeof MediaRecorder === 'undefined') {
      reject(new Error('MediaRecorder is not supported. Use Chrome or Edge.'));
      return;
    }

    const options = liveGetRecorderOptions();

    if (options === null) {
      reject(new Error('MediaRecorder is not supported.'));
      return;
    }

    let recorder;

    try {
      recorder = new MediaRecorder(stream, options);
    } catch (err) {
      reject(err);
      return;
    }

    const chunks = [];

    recorder.ondataavailable = function(event) {
      if (event.data && event.data.size > 0) {
        chunks.push(event.data);
      }
    };

    recorder.onerror = function(event) {
      reject(event.error || new Error('MediaRecorder error'));
    };

    recorder.onstop = function() {
      const mimeType = recorder.mimeType || 'video/webm';
      const blob = new Blob(chunks, { type: mimeType });
      resolve(blob);
    };

    recorder.start(80);

    setTimeout(function() {
      if (recorder.state !== 'inactive') {
        recorder.stop();
      }
    }, durationMs);
  });
}


// ===============================
// Backend Request
// ===============================

async function sendLiveChunkToBackend(blob) {
  const fd = new FormData();

  const file = new File(
    [blob],
    'live_sign_' + Date.now() + '.webm',
    { type: blob.type || 'video/webm' }
  );

  fd.append('video', file);

  // Keep top_k small for speed.
  fd.append('top_k', '5');
  fd.append('check_hand', 'true');

  const response = await fetch(LIVE_ENDPOINT, {
    method: 'POST',
    body: fd
  });

  return await response.json();
}


// ===============================
// Prediction Extractor
// ===============================

function normalizeGloss(gloss) {
  return String(gloss || '')
    .trim()
    .toUpperCase();
}

function cleanDisplayWord(word) {
  word = String(word || '').trim();

  if (!word) {
    return '';
  }

  return word
    .replace(/_/g, ' ')
    .replace(/\s+/g, ' ');
}

function extractPrediction(data) {
  if (!data || !data.ok || !data.result) {
    return null;
  }

  const result = data.result;

  let gloss =
    result.gloss ||
    result.label ||
    result.predicted_label ||
    '';

  let text =
    result.text ||
    result.clean_text ||
    result.word ||
    result.predicted_text ||
    '';

  let confidence =
    result.confidence ||
    result.probability ||
    result.score ||
    null;

  if ((!gloss || confidence === null) && result.top_k && result.top_k.length > 0) {
    const top1 = result.top_k[0];

    if (!gloss) {
      gloss = top1.gloss || top1.label || '';
    }

    if (!text) {
      text = top1.text || top1.word || top1.gloss || '';
    }

    if (confidence === null || confidence === undefined) {
      confidence = top1.confidence || top1.probability || top1.score || 0;
    }
  }

  gloss = normalizeGloss(gloss);
  text = cleanDisplayWord(text || gloss);
  confidence = Number(confidence || 0);

  if (!gloss && text) {
    gloss = normalizeGloss(text);
  }

  if (!gloss) {
    return null;
  }

  return {
    gloss: gloss,
    text: text,
    confidence: confidence,
    raw: result
  };
}


// ===============================
// Sentence Builder
// ===============================

const GLOSS_TO_WORD = {
  'MYSELF': 'I',

  'WANT1': 'want',
  'WANT': 'want',

  'EAT1': 'eat',
  'EAT': 'eat',

  'BREAKFAST1': 'breakfast',
  'BREAKFAST': 'breakfast',

  'LUNCH1': 'lunch',
  'LUNCH': 'lunch',

  'DINNER1': 'dinner',
  'DINNER': 'dinner',

  'APPLE': 'apple',

  'CANDY2': 'candy',
  'CANDY': 'candy',

  'MEAT1': 'meat',
  'MEAT': 'meat',

  'TOMATO': 'tomato',

  'HOSPITAL1': 'hospital',
  'HOSPITAL': 'hospital',

  'BANDAGE': 'bandage',

  'FINE1': 'fine',
  'FINE': 'fine',

  'CONFUSED1': 'confused',
  'CONFUSED': 'confused',

  'SHOCKED': 'shocked',

  'BABY2': 'baby',
  'BABY': 'baby',

  'PATIENT2': 'patient',
  'PATIENT': 'patient',

  'DEAF1': 'deaf',
  'DEAF': 'deaf',

  'HARDOFHEARING': 'hard of hearing',
  'VOICE': 'voice',

  'HOUSE': 'home',
  'ADDRESS': 'address',

  'BACKPACK1': 'backpack',
  'BACKPACK': 'backpack',

  'JACKET3': 'jacket',
  'JACKET': 'jacket',

  'SCARF2': 'scarf',
  'SCARF': 'scarf',

  'SWEATER2': 'sweater',
  'SWEATER': 'sweater',

  'BELT1': 'belt',
  'BELT': 'belt',

  'JEWELRY': 'jewelry',

  'STOP': 'stop',
  'CLOSE': 'close',

  'MOVIE1': 'movie',
  'MOVIE': 'movie',

  'NIGHT1': 'night',
  'NIGHT': 'night',

  'PARTY1': 'party',
  'PARTY': 'party',

  'THEY1': 'they',
  'THEY': 'they',

  'TYPE1': 'type',
  'TYPE': 'type',

  'EDIT1': 'edit',
  'EDIT': 'edit',

  'RESEARCH1': 'research',
  'RESEARCH': 'research',

  'TEACH1': 'teach',
  'TEACH': 'teach',

  'UNDERGRADUATE': 'undergraduates'
};

const EXACT_SENTENCE_MAP = {
  'MYSELF WANT1 BREAKFAST1': 'I want breakfast.',
  'MYSELF WANT1 LUNCH1': 'I want lunch.',
  'MYSELF WANT1 DINNER1': 'I want dinner.',
  'MYSELF WANT1 APPLE': 'I want an apple.',
  'MYSELF WANT1 CANDY2': 'I want candy.',

  'MYSELF EAT1 APPLE': 'I eat apple.',
  'MYSELF EAT1 MEAT1': 'I eat meat.',
  'MYSELF EAT1 TOMATO': 'I eat tomato.',
  'MYSELF EAT1 LUNCH1': 'I eat lunch.',
  'MYSELF EAT1 DINNER1': 'I eat dinner.',

  'MYSELF WANT1 HOSPITAL1': 'I need hospital.',
  'MYSELF WANT1 BANDAGE': 'I need a bandage.',
  'MYSELF FINE1': 'I am fine.',
  'MYSELF CONFUSED1': 'I am confused.',
  'MYSELF SHOCKED': 'I am shocked.',

  'BABY2 FINE1': 'The baby is fine.',
  'PATIENT2 CONFUSED1': 'The patient is confused.',
  'THEY1 DEAF1': 'They are deaf.',
  'THEY1 HARDOFHEARING': 'They are hard of hearing.',
  'MYSELF WANT1 VOICE': 'I need voice.',

  'MYSELF WANT1 HOUSE': 'I want to go home.',
  'MYSELF WANT1 ADDRESS': 'I need the address.',
  'MYSELF WANT1 BACKPACK1': 'I want my backpack.',
  'MYSELF WANT1 JACKET3': 'I want my jacket.',
  'MYSELF WANT1 SCARF2': 'I want a scarf.',
  'MYSELF WANT1 SWEATER2': 'I want a sweater.',
  'MYSELF WANT1 BELT1': 'I want a belt.',
  'MYSELF WANT1 JEWELRY': 'I want jewelry.',

  'STOP': 'Stop.',
  'CLOSE': 'Close it.',

  'MYSELF WANT1 MOVIE1': 'I want a movie.',
  'MOVIE1 NIGHT1': 'Movie night.',
  'MYSELF WANT1 PARTY1': 'I want a party.',
  'PARTY1 NIGHT1': 'Party night.',

  'THEY1 WANT1 DINNER1': 'They want dinner.',
  'THEY1 WANT1 HOUSE': 'They want to go home.',

  'MYSELF TYPE1': 'I type.',
  'MYSELF EDIT1': 'I edit.',
  'MYSELF RESEARCH1': 'I do research.',
  'MYSELF TEACH1 UNDERGRADUATE': 'I teach undergraduates.'
};

function canonicalGloss(gloss) {
  gloss = normalizeGloss(gloss);

  const aliases = {
    'WANT': 'WANT1',
    'EAT': 'EAT1',
    'BREAKFAST': 'BREAKFAST1',
    'LUNCH': 'LUNCH1',
    'DINNER': 'DINNER1',
    'CANDY': 'CANDY2',
    'MEAT': 'MEAT1',
    'HOSPITAL': 'HOSPITAL1',
    'FINE': 'FINE1',
    'CONFUSED': 'CONFUSED1',
    'BABY': 'BABY2',
    'PATIENT': 'PATIENT2',
    'DEAF': 'DEAF1',
    'BACKPACK': 'BACKPACK1',
    'JACKET': 'JACKET3',
    'SCARF': 'SCARF2',
    'SWEATER': 'SWEATER2',
    'BELT': 'BELT1',
    'MOVIE': 'MOVIE1',
    'NIGHT': 'NIGHT1',
    'PARTY': 'PARTY1',
    'THEY': 'THEY1',
    'TYPE': 'TYPE1',
    'EDIT': 'EDIT1',
    'RESEARCH': 'RESEARCH1',
    'TEACH': 'TEACH1'
  };

  return aliases[gloss] || gloss;
}

function capitalizeSentence(text) {
  text = String(text || '').trim();

  if (!text) {
    return '---';
  }

  text = text.charAt(0).toUpperCase() + text.slice(1);

  if (!/[.!?]$/.test(text)) {
    text += '.';
  }

  return text;
}

function buildEnglishFromGlosses(glosses) {
  if (!glosses || glosses.length === 0) {
    return '---';
  }

  const canonical = glosses.map(canonicalGloss);
  const key = canonical.join(' ');

  if (EXACT_SENTENCE_MAP[key]) {
    return EXACT_SENTENCE_MAP[key];
  }

  let sentence = canonical
    .map(function(g) {
      return GLOSS_TO_WORD[g] || g.toLowerCase();
    })
    .join(' ')
    .replace(/\s+/g, ' ')
    .trim();

  sentence = sentence
    .replace(/^I want hospital$/i, 'I need hospital')
    .replace(/^I want bandage$/i, 'I need a bandage')
    .replace(/^I want address$/i, 'I need the address')
    .replace(/^I want voice$/i, 'I need voice')
    .replace(/^I want home$/i, 'I want to go home')
    .replace(/^they want home$/i, 'They want to go home')
    .replace(/^I fine$/i, 'I am fine')
    .replace(/^I confused$/i, 'I am confused')
    .replace(/^I shocked$/i, 'I am shocked')
    .replace(/^baby fine$/i, 'The baby is fine')
    .replace(/^patient confused$/i, 'The patient is confused')
    .replace(/^they deaf$/i, 'They are deaf')
    .replace(/^they hard of hearing$/i, 'They are hard of hearing')
    .replace(/^movie night$/i, 'Movie night')
    .replace(/^party night$/i, 'Party night')
    .replace(/^I research$/i, 'I do research');

  return capitalizeSentence(sentence);
}

function updateLiveSentenceUI() {
  if (liveGlosses.length === 0) {
    liveSetText('liveGlossSequence', '---');
    liveSetText('liveEnglishSentence', '---');
    return;
  }

  liveSetText('liveGlossSequence', liveGlosses.join(' + '));
  liveSetText('liveEnglishSentence', buildEnglishFromGlosses(liveGlosses));
}

function acceptPendingCandidateByRelease() {
  if (!liveCandidatePred) {
    return false;
  }

  if (liveCandidatePred.confidence < LIVE_RELEASE_CONFIRM_CONFIDENCE) {
    liveSetText(
      'liveStatus',
      'Hand disappeared, but confidence was too low: ' + liveCandidatePred.gloss
    );

    liveCandidateGloss = '';
    liveCandidateCount = 0;
    liveCandidatePred = null;

    return false;
  }

  acceptPrediction(liveCandidatePred);

  liveSetText(
    'liveStatus',
    'Accepted by hand release: ' + liveCandidatePred.gloss
  );

  return true;
}
// ===============================
// Acceptance Logic
// ===============================

function shouldAcceptPrediction(pred) {
  if (!pred || pred.confidence < LIVE_MIN_CONFIDENCE) {
    return false;
  }

  const now = Date.now();

  const currentGloss = canonicalGloss(pred.gloss);
  const lastAcceptedGloss = canonicalGloss(liveLastAcceptedGloss);
  const candidateGloss = canonicalGloss(liveCandidateGloss);

  // ما نكررش نفس الكلمة المقبولة قريب
  if (
    currentGloss === lastAcceptedGloss &&
    now - liveLastAcceptedTime < LIVE_REPEAT_COOLDOWN_MS
  ) {
    liveSetText(
      'liveStatus',
      'Same accepted sign ignored: ' + currentGloss
    );

    return false;
  }

  // أول توقع للحركة: نخزنه pending بس، لسه مانعتمدوش
  if (!candidateGloss) {
    liveCandidateGloss = currentGloss;
    liveCandidateCount = 1;
    liveCandidatePred = pred;

    liveSetText(
      'liveStatus',
      'Candidate: ' + currentGloss + ' | Release your hand to confirm'
    );

    return false;
  }

  // لو نفس الحركة لسه مستمرة
  if (currentGloss === candidateGloss) {
    liveCandidateCount += 1;

    // نحدّث أفضل prediction لو الثقة أعلى
    if (!liveCandidatePred || pred.confidence > liveCandidatePred.confidence) {
      liveCandidatePred = pred;
    }

    liveSetText(
      'liveStatus',
      'Holding: ' + currentGloss +
      ' | confidence ' + pred.confidence.toFixed(3) +
      ' | release hand to confirm'
    );

    // لو هو مصر يفضل عامل نفس الحركة، نعتمدها فقط لو الثقة بقت عالية
    if (
      liveCandidateCount >= LIVE_STABLE_COUNT_REQUIRED &&
      pred.confidence >= LIVE_HOLD_CONFIRM_CONFIDENCE
    ) {
      return true;
    }

    return false;
  }

  // لو الحركة اتغيرت قبل ما الإيد تختفي
  // يبقى التوقع القديم مرفوض ونبدأ Candidate جديد
  liveSetText(
    'liveStatus',
    'Movement changed. Rejected: ' + candidateGloss + ' | New candidate: ' + currentGloss
  );

  liveCandidateGloss = currentGloss;
  liveCandidateCount = 1;
  liveCandidatePred = pred;

  return false;
}

function acceptPrediction(pred) {
  const gloss = canonicalGloss(pred.gloss);

  liveGlosses.push(gloss);
  liveWords.push(pred.text);

  liveLastAcceptedGloss = gloss;
  liveLastAcceptedTime = Date.now();

  liveCandidateGloss = '';
  liveCandidateCount = 0;
  liveCandidatePred = null;

  updateLiveSentenceUI();
}


// ===============================
// Live Main Loop
// ===============================

async function runOneLiveCycle() {
  if (!liveIsRunning || liveIsProcessing || !liveStream) {
    return;
  }

  liveIsProcessing = true;

  try {
    liveSetText('liveStatus', 'Recording short chunk...');

    const blob = await recordLiveChunk(liveStream, LIVE_CHUNK_MS);

    if (!liveIsRunning) {
      liveIsProcessing = false;
      return;
    }

    liveSetText('liveStatus', 'Processing sign...');

    const data = await sendLiveChunkToBackend(blob);

    const noHand =
      data &&
      data.ok &&
      data.result &&
      (
        data.result.no_hand === true ||
        data.result.hand_visible === false
      );

    if (noHand) {
      liveSetText('liveCurrentWord', '---');
      liveSetText('liveConfidence', 'Confidence: ---');

      liveSetJSON('liveRawResult', {
        ok: data.ok,
        no_hand: true,
        pending_candidate: liveCandidatePred ? {
          gloss: liveCandidatePred.gloss,
          confidence: liveCandidatePred.confidence
        } : null
      });

      if (liveCandidatePred) {
        const acceptedGloss = liveCandidatePred.gloss;

        acceptPrediction(liveCandidatePred);

        liveSetText(
          'liveStatus',
          'Hand disappeared. Accepted: ' + acceptedGloss
        );

        liveIsProcessing = false;

        if (liveIsRunning) {
          liveLoopTimer = setTimeout(runOneLiveCycle, LIVE_AFTER_ACCEPT_GAP_MS);
        }

        return;
      }

      liveSetText('liveStatus', 'Waiting for hand...');

      liveIsProcessing = false;

      if (liveIsRunning) {
        liveLoopTimer = setTimeout(runOneLiveCycle, LIVE_NO_HAND_GAP_MS);
      }

      return;
    }

    const pred = extractPrediction(data);

    liveSetJSON('liveRawResult', {
      ok: data.ok,
      current_prediction: pred ? {
        gloss: pred.gloss,
        confidence: pred.confidence
      } : null,
      top_k: data.result && data.result.top_k
        ? data.result.top_k.slice(0, 5)
        : null
    });

    if (pred) {
      liveSetText('liveCurrentWord', pred.gloss);
      liveSetText('liveConfidence', 'Confidence: ' + pred.confidence.toFixed(3));

      if (shouldAcceptPrediction(pred)) {
        acceptPrediction(pred);

        liveSetText(
          'liveStatus',
          'Accepted: ' + pred.gloss + ' | Keep signing...'
        );

      } else {
        if (pred.confidence < LIVE_MIN_CONFIDENCE) {
          liveSetText(
            'liveStatus',
            'Low confidence ignored: ' + pred.gloss
          );
        } else {
          liveSetText(
            'liveStatus',
            'Waiting for stable prediction: ' + pred.gloss
          );
        }
      }

    } else {
      liveSetText('liveCurrentWord', '---');
      liveSetText('liveConfidence', 'Confidence: ---');

      if (liveCandidatePred) {
        acceptPendingCandidateByRelease();
      } else {
        liveSetText('liveStatus', 'No clear sign detected.');
      }
    }

  } catch (err) {
    console.error('Live cycle error:', err);

    liveSetText(
      'liveStatus',
      'Live error: ' + (err.message || String(err))
    );
  }

  liveIsProcessing = false;

  if (liveIsRunning) {
    liveLoopTimer = setTimeout(runOneLiveCycle, LIVE_LOOP_GAP_MS);
  }
}

function startLiveTranslation() {
  if (!liveStream) {
    alert('Open camera first.');
    return;
  }

  if (liveIsRunning) {
    return;
  }

  liveIsRunning = true;

  liveSetButton('liveStartBtn', true);
  liveSetButton('liveStopBtn', false);

  liveSetText(
    'liveStatus',
    'Live translation started. Perform one sign every 1–1.5 seconds.'
  );

  runOneLiveCycle();
}

function stopLiveTranslation() {
  liveIsRunning = false;
  liveIsProcessing = false;

  if (liveLoopTimer) {
    clearTimeout(liveLoopTimer);
    liveLoopTimer = null;
  }

  liveSetButton('liveStartBtn', false);
  liveSetButton('liveStopBtn', true);

  liveSetText('liveStatus', 'Live translation stopped.');
}

function clearLiveSentence() {
  liveGlosses = [];
  liveWords = [];

  liveLastAcceptedGloss = '';
  liveLastAcceptedTime = 0;

  liveCandidateGloss = '';
  liveCandidateCount = 0;
  liveCandidatePred = null;


  liveSetText('liveCurrentWord', '---');
  liveSetText('liveConfidence', 'Confidence: ---');
  liveSetText('liveGlossSequence', '---');
  liveSetText('liveEnglishSentence', '---');
  liveSetJSON('liveRawResult', '---');

  liveSetText('liveStatus', 'Sentence cleared.');
}
// ===============================
// Live Sentence → TTS
// ===============================

function liveSetAudio(id, url) {
  const audio = document.getElementById(id);

  if (!audio) {
    return;
  }

  if (url) {
    audio.src = url;
    audio.style.display = 'block';
    audio.load();
  } else {
    audio.pause();
    audio.removeAttribute('src');
    audio.style.display = 'none';
  }
}

async function speakLiveSentence() {
  const sentenceBox = document.getElementById('liveEnglishSentence');

  if (!sentenceBox) {
    alert('Live sentence box not found.');
    return;
  }

  const sentence = sentenceBox.textContent.trim();

  if (!sentence || sentence === '---') {
    alert('No live sentence to speak yet.');
    return;
  }

  liveSetText('liveStatus', 'Generating speech for live sentence...');
  liveSetAudio('liveTtsAudio', null);

  const fd = new FormData();
  fd.append('text', sentence);

  try {
    const response = await fetch('/api/tts', {
      method: 'POST',
      body: fd
    });

    const data = await response.json();

    liveSetJSON('liveRawResult', {
      tts_request_text: sentence,
      tts_response: data
    });

    if (data.ok && data.result && data.result.audio_url) {
      liveSetAudio('liveTtsAudio', data.result.audio_url);

      const audio = document.getElementById('liveTtsAudio');

      if (audio) {
        await audio.play();
      }

      liveSetText('liveStatus', 'Speech generated and playing.');
    } else {
      liveSetText('liveStatus', 'TTS error.');
      alert('TTS failed. Check Raw Live Result.');
    }

  } catch (err) {
    liveSetText('liveStatus', 'TTS error: ' + err.message);

    liveSetJSON('liveRawResult', {
      ok: false,
      error: err.message
    });
  }
}

window.addEventListener('beforeunload', function() {
  stopLiveTranslation();
  closeLiveCamera();
});