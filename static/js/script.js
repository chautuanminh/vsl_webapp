document.addEventListener('DOMContentLoaded', () => {
    // --- Live Stream Controls ---
    const videoFeed = document.getElementById('video-feed');
    const confSlider = document.getElementById('conf-slider');
    const confVal = document.getElementById('conf-val');

    let debounceTimer;

    confSlider.addEventListener('input', (e) => {
        const value = e.target.value;
        confVal.textContent = value;

        // Debounce updating the stream to avoid flickering
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            // Update src with new timestamp to force reload and new conf
            videoFeed.src = `/video_feed?conf=${value}&t=${new Date().getTime()}`;
        }, 300);
    });

    // --- Image Upload ---
    const uploadForm = document.getElementById('upload-form');
    const uploadResult = document.getElementById('upload-result');
    const resultImage = document.getElementById('result-image');
    const uploadConf = document.getElementById('upload-conf');

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const fileInput = document.getElementById('image-input');
        if (fileInput.files.length === 0) return;

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('conf', confSlider.value); // Use current slider value

        try {
            const response = await fetch('/detect_image', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Detection failed');
            }

            const data = await response.json();

            // Display result
            resultImage.src = data.image;
            uploadConf.textContent = data.avg_conf.toFixed(2);
            uploadResult.classList.remove('hidden');

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during detection.');
        }
    });
});
