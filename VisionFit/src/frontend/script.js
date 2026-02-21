// frontend/script.js
const fileInput = document.getElementById('fileInput');
const startCam = document.getElementById('startCam');
const captureBtn = document.getElementById('captureBtn');
const stopCam = document.getElementById('stopCam');
const video = document.getElementById('video');
const canvas = document.getElementById('preview');
const ctx = canvas.getContext('2d');
const resultDiv = document.getElementById('result');

let stream = null;

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      canvas.toBlob((blob) => sendBlob(blob), 'image/jpeg', 0.95);
    }
    img.src = reader.result;
  };
  reader.readAsDataURL(file);
});

startCam.addEventListener('click', async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({video: true, audio: false});
    video.srcObject = stream;
    video.style.display = 'block';
    captureBtn.disabled = false;
    stopCam.disabled = false;
  } catch (err) {
    alert('Could not start camera: ' + err.message);
  }
});

stopCam.addEventListener('click', () => {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
    video.style.display = 'none';
    captureBtn.disabled = true;
    stopCam.disabled = true;
  }
});

captureBtn.addEventListener('click', () => {
  if (!stream) return;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0);
  canvas.toBlob((blob) => sendBlob(blob), 'image/jpeg', 0.95);
});

async function sendBlob(blob) {
  resultDiv.textContent = 'Sending...';
  const form = new FormData();
  form.append('image', blob, 'capture.jpg');

  try {
    const resp = await fetch('/predict', { method: 'POST', body: form });
    if (!resp.ok) {
      const txt = await resp.text();
      throw new Error('Server error: ' + txt);
    }
    const data = await resp.json();
    resultDiv.innerHTML = `
      <strong>Face shape:</strong> ${data.face_shape} (${(data.confidence*100).toFixed(1)}%)<br>
      <strong>Recommendations:</strong> ${data.recommendations.join(', ')}
    `;
  } catch (err) {
    resultDiv.textContent = 'Error: ' + err.message;
  }
}


