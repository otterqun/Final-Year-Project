const fileInput = document.getElementById('fileInput');
const startCam = document.getElementById('startCam');
const captureBtn = document.getElementById('captureBtn');
const stopCam = document.getElementById('stopCam');
const video = document.getElementById('video');
const canvas = document.getElementById('preview');
const ctx = canvas.getContext('2d');
const resultDiv = document.getElementById('result');

let stream = null;
let intervalId = null;

// --- Handle Image Upload ---
fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    const img = new Image();
    img.onload = async () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      const blob = await new Promise((r) => canvas.toBlob(r, 'image/jpeg', 0.95));
      sendBlob(blob);
    };
    img.src = reader.result;
  };
  reader.readAsDataURL(file);
});

// --- Start Webcam ---
startCam.addEventListener('click', async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.style.display = 'block';
    captureBtn.disabled = false;
    stopCam.disabled = false;

    // Send frames every 2 seconds
    intervalId = setInterval(captureAndSendFrame, 2000);
  } catch (err) {
    alert('Could not start camera: ' + err.message);
  }
});

// --- Stop Webcam ---
stopCam.addEventListener('click', () => {
  if (intervalId) clearInterval(intervalId);
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.style.display = 'none';
  captureBtn.disabled = true;
  stopCam.disabled = true;
  resultDiv.textContent = '';
  ctx.clearRect(0, 0, canvas.width, canvas.height);
});

// --- Manual Capture Button ---
captureBtn.addEventListener('click', captureAndSendFrame);

function captureAndSendFrame() {
  if (!stream) return;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0);
  canvas.toBlob((blob) => sendBlob(blob), 'image/jpeg', 0.95);
}

// --- Send Image to Backend ---
async function sendBlob(blob) {
  const form = new FormData();
  form.append('image', blob, 'capture.jpg');

  try {
    const resp = await fetch('/predict', { method: 'POST', body: form });
    const data = await resp.json();

    if (data.error) {
      resultDiv.textContent = data.error;
      return;
    }

    // Draw bounding box
    const { x, y, w, h } = data.bounding_box;
    ctx.lineWidth = 3;
    ctx.strokeStyle = 'lime';
    ctx.strokeRect(x, y, w, h);

    resultDiv.innerHTML = `
      <strong>Face shape:</strong> ${data.face_shape} (${(data.confidence * 100).toFixed(1)}%)<br>
      <strong>Recommendations:</strong> ${data.recommendations.join(', ')}
    `;
  } catch (err) {
    resultDiv.textContent = 'Error: ' + err.message;
  }
}
