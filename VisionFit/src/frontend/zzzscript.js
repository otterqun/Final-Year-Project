// ===============================
// 1. ELEMENT SELECTION
// ===============================
const fileInput = document.getElementById('fileInput');
const startCam = document.getElementById('startCam');
const captureBtn = document.getElementById('captureBtn');
const stopCam = document.getElementById('stopCam');
const video = document.getElementById('video');
const canvas = document.getElementById('preview'); // Canvas untuk input preview
const ctx = canvas.getContext('2d');
const resultDiv = document.getElementById('result'); // Ini Div Kanan (Output)

// UI Logic Elements
const chooseUpload = document.getElementById('chooseUpload');
const chooseCamera = document.getElementById('chooseCamera');
const uploadSection = document.getElementById('uploadSection');
const cameraSection = document.getElementById('cameraSection');

let stream = null;

// ===============================
// 2. UI SWITCHING LOGIC (Upload vs Camera)
// ===============================

// Default: Hide all inputs
cameraSection.style.display = "none";
uploadSection.style.display = "none";
resultDiv.innerHTML = `<p style="color: #888; text-align:center; margin-top:50px;">Waiting for input...</p>`;

chooseUpload.addEventListener('click', () => {
  uploadSection.style.display = "block";
  cameraSection.style.display = "none";
  stopCameraStream(); // Matikan kamera jika tengah on
  clearResult();
});

chooseCamera.addEventListener('click', () => {
  cameraSection.style.display = "block";
  uploadSection.style.display = "none";
  clearResult();
});

function clearResult() {
    resultDiv.innerHTML = `<p style="color: #888; text-align:center; margin-top:50px;">Waiting for input...</p>`;
    canvas.style.display = 'none'; // Hide preview canvas awal-awal
}

// ===============================
// 3. UPLOAD HANDLER
// ===============================
fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = () => {
    const img = new Image();
    img.onload = () => {
      // Draw image to canvas (Left Side Preview)
      canvas.style.display = "block";
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      // Send to Backend
      canvas.toBlob((blob) => sendBlob(blob), "image/jpeg", 0.95);
    };
    img.src = reader.result;
  };
  reader.readAsDataURL(file);
});

// ===============================
// 4. CAMERA HANDLER
// ===============================
startCam.addEventListener('click', async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    video.style.display = 'block';
    captureBtn.disabled = false;
    stopCam.disabled = false;
    canvas.style.display = 'none'; // Hide static canvas bila video on
  } catch (err) {
    alert('Could not start camera: ' + err.message);
  }
});

stopCam.addEventListener('click', stopCameraStream);

function stopCameraStream() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
    video.style.display = 'none';
    captureBtn.disabled = true;
    stopCam.disabled = true;
  }
}

captureBtn.addEventListener('click', () => {
  if (!stream) return;

  // 1. Pause video untuk effect "Freeze"
  video.pause();

  // 2. Draw frame ke canvas (Hidden processing)
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // 3. Send to Backend
  canvas.toBlob((blob) => sendBlob(blob), "image/jpeg", 0.95);
  
  // Optional: Resume video after 2 seconds?
  // setTimeout(() => video.play(), 2000);
});

// ===============================
// 5. BACKEND COMMUNICATION (The Core Logic)
// ===============================
async function sendBlob(blob) {
  
  // A. STEP 1: Show Loading State di Grid Kanan
  resultDiv.innerHTML = `
    <div style="text-align:center; margin-top:50px;">
        <div class="loader" style="border: 4px solid #f3f3f3; border-top: 4px solid #00ffd5; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto;"></div>
        <p style="color: #00ffd5; margin-top:15px;">Scanning Face & Analyzing...</p>
    </div>
    <style>@keyframes spin {0% {transform: rotate(0deg);} 100% {transform: rotate(360deg);}}</style>
  `;

  const form = new FormData();
  form.append('image', blob, 'capture.jpg');

  try {
    // B. STEP 2: Fetch Data
    const resp = await fetch('/predict', { method: 'POST', body: form });
    
    if (!resp.ok) {
      const txt = await resp.text();
      throw new Error('Server error: ' + txt);
    }
    
    const data = await resp.json();

    // C. STEP 3: Check Face Detection (Backend Logic)
if (data.error) {
    resultDiv.innerHTML = `
        <div style="...">
            <h3>⚠️ No Face Detected</h3>
            <p>Please adjust lighting or face the camera directly.</p>
             ...
        </div>
    `;
    return;
}

    // D. STEP 4: Render Result (The "Grid Kedua" Logic)
    // Pastikan backend hantar 'cropped_image' (Base64 string)
    
    const croppedImageSrc = data.cropped_image ? `data:image/jpeg;base64,${data.cropped_image}` : 'placeholder.jpg';

    // PAPARAN BARU
resultDiv.innerHTML = `
  <div class="result-card" style="text-align: center;">
    
    <div class="crop-zone" style="margin-bottom: 20px;">
        <p style="color:#aaa; font-size:0.9rem; margin-bottom:10px; ">Detected Face</p>
        <img src="${data.cropped_image ? 'data:image/jpeg;base64,' + data.cropped_image : ''}" 
             style="width: 150px; height: 150px; object-fit: cover; border-radius: 25px; border: 3px solid #00ffd5; box-shadow: 0 0 15px rgba(0,255,213,0.5);">
    </div>

    <h2 style="color: #00ffd5; text-transform: uppercase; font-size: 2rem; margin-bottom: 5px;">
        ${data.face_shape}
    </h2>
    <p style="color: #ccc; margin-top:20px; margin-bottom: 20px; font-style:Arial ; max-width: 80%; margin-left: auto; margin-right: auto;">
        "${data.recommendations.traits}"
    </p>
    

    <div class="recommendations" style="text-align: left; background: rgba(0,0,0,0.2); padding: 15px; border-radius: 10px;">
        <h4 style="color: #ccc; border-bottom: 1px solid #444; padding-bottom: 5px; margin-bottom: 10px;">Recommended Frames:</h4>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); gap: 10px;">
            
            ${data.recommendations.frames.map(rec => `
                <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 10px; text-align: center; border: 1px solid rgba(255,255,255,0.1); transition: transform 0.2s;">
                    
                    <div style="width: 100%; height: 80px; background: #ffffff; border-radius: 6px; margin-bottom: 8px; display: flex; align-items: center; justify-content: center; overflow: hidden;">
                        <img src="${rec.image}" alt="${rec.name}" 
                            style="max-width: 100%; max-height: 100%; object-fit: contain; display: block;">
                    </div>
                    
                    <p style="color: #00ffd5; font-size: 0.7rem; margin: 0;  text-transform: uppercase; letter-spacing: 0.5px;">
                        ${rec.name}
                    </p>
                </div>
            `).join('')}

        </div>
    </div>
  </div>
`;

  } catch (err) {
    resultDiv.innerHTML = `<p style="color: #ff006e; text-align:center;">Error: ${err.message}</p>`;
  }
}

// <p style="color: #fff; margin-bottom: 20px;">Confidence: ${(data.confidence * 100).toFixed(1)}%</p>