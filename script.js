//SM-Note-Run server
//(venv) C:\0.data\4.SM-WSpace\6B.Python\1.Create_Video_From_Readernook_Story\application>python server.py

// 🔹 Dummy Caption Data for Preview
let dummyCaptionsData = [
    { "word": "This", "start": 0.0, "end": 0.22 },
    { "word": "is", "start": 0.22, "end": 0.36 },
    { "word": "a", "start": 0.36, "end": 0.48 },
    { "word": "caption", "start": 0.48, "end": 0.72 },
    { "word": "preview", "start": 0.72, "end": 1.12 }
];

const textColors = [
    "#222222",  
    "#9f2f16",  
    "#85540d",  
    "#46850d",  
    "#0d5a85",  
    "#5f0d85"   
];

const bgColors = [
    "#e9d7f2",  
    "#d7eaf2",  
    "#adf3e2",  
    "#d2f3ad",  
    "#f3efad",  
    "#f3c7ad"   
];

const fontSizes = ["1.9em", "1.8em", "2em", "2.5em", "2.3em"];
const angles = ["angle1", "angle2", "angle3", "angle4", "angle5", "angle6"];

const wordEditor = document.getElementById("word-editor");
const saveWordChanges = document.getElementById("save-word-changes");
let wordTimestamps = [];


let currentOverlayText = "";

let currentStayingHeading = "";
let stayingListItems = [];





//let currentCaptionIndex = 0;  // Track the index of the caption being displayed
let captionsData = [];
let currentCaption = "";

let captionWordLimit = 5;  // Default number of words per block
let currentBlockStart = 0; // Track where the current caption block starts
let lastSpokenWordIndex = -1; // Last word index processed
let lastCaptionUpdateTime = 0; // Time when captions were last updated

let previewTime = 0.0;
let previewInterval;

// Load full_text and word_timestamps (for captions)
//fetch('temp/word_timestamps.json')
fetch("http://localhost:5000/get_word_timestamps")
    .then(response => response.json())
    .then(data => {
        captionsData = data;
        wordTimestamps = data;
        renderWordEditor();          
    })
    .catch(error => console.error("Error loading captions data:", error));

const video = document.getElementById("video");
const playPauseBtn = document.getElementById("playPauseBtn");
const restartBtn = document.getElementById("restartBtn");
const videoTimeDisplay = document.getElementById("video-time");

const captionStyleDropdown = document.getElementById("captionStyle");
const captionPreview = document.getElementById("caption-preview");

const subscribeGif = document.getElementById("subscribe-gif");

video.volume = 1.0;  // Set default volume to max

// 🔹 Hide Default Video Controls
video.removeAttribute("controls");

// 🔹 Play/Pause Button Functionality
playPauseBtn.addEventListener("click", () => {
    if (video.paused) {
        video.play();
        playPauseBtn.innerHTML = "⏸ Pause";
    } else {
        video.pause();
        playPauseBtn.innerHTML = "▶ Play";
    }
});

// 🔹 Restart Button Functionality
restartBtn.addEventListener("click", () => {
    video.currentTime = 0;
    video.play();
    playPauseBtn.innerHTML = "⏸ Pause";  // Change button to "Pause" when restarted

    //currentCaptionIndex = 0;  // Reset caption tracking
    captions.innerHTML = "";  // Clear previous captions
    currentBlockStart = 0
});

// 🔹 Format Time Function (Convert Seconds to mm:ss)
function formatTime(time) {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

// // 🔹 Update Time Display
// video.addEventListener("timeupdate", () => {
//     videoTimeDisplay.innerHTML = `${formatTime(video.currentTime)} / ${formatTime(video.duration)}`;
// });

// 🔹 Update Total Duration When Metadata Loads
video.addEventListener("loadedmetadata", () => {
    videoTimeDisplay.innerHTML = `00:00 / ${formatTime(video.duration)}`;
});

const overlay = document.getElementById("overlayText");


const overlayContainer = document.getElementById("overlayContainer");
const stayingHeading = document.createElement("div"); // Element for staying headings
stayingHeading.classList.add("staying-heading");
overlayContainer.appendChild(stayingHeading);

const stayingListContainer = document.createElement("div"); // Container for staying list items
stayingListContainer.classList.add("staying-list-container");
overlayContainer.appendChild(stayingListContainer);


const captions = document.getElementById("captions");

const captionInput = document.getElementById("captionLength");

// Update caption word limit dynamically from user input
captionInput.addEventListener("input", () => {
    captionWordLimit = parseInt(captionInput.value, 10) || 5;
});

let selectedStyle = "style1";  // Default caption style
// Change Caption Style Based on Selection
captionStyleDropdown.addEventListener("change", () => {
    selectedStyle = captionStyleDropdown.value;

    // Remove old styles
    captions.className = "captions-text";
    captionPreview.className = "preview-captions-text";

    // Apply new style
    captions.classList.add(selectedStyle);
    captionPreview.classList.add(selectedStyle);
});

// Update caption word limit dynamically from user input
captionInput.addEventListener("input", () => {
    captionWordLimit = parseInt(captionInput.value, 10) || 5;
});

let videoDuration = video.duration;

// Listen for video time updates
video.addEventListener("timeupdate", () => {
    const currentTime = video.currentTime;
    

    if (currentTime >= 30 && currentTime <= 35) {
        // Show GIF 30 seconds after start (for 5 seconds)
        subscribeGif.classList.add("show-gif");
        subscribeGif.classList.remove("hidden");
    } else if (videoDuration - currentTime <= 30 && videoDuration - currentTime >= 25) {
        // Show GIF 30 seconds before end (for 5 seconds)
        subscribeGif.classList.add("show-gif");
        subscribeGif.classList.remove("hidden");
    } else {
        // Hide otherwise
        subscribeGif.classList.remove("show-gif");
        subscribeGif.classList.add("hidden");
    }

    /** 🔹 2. Display Captions in Blocks & Maintain Them During Pauses **/
    videoTimeDisplay.innerHTML = `${formatTime(video.currentTime)} / ${formatTime(video.duration)}`;

    let currentIndex = captionsData.findIndex(word => currentTime >= word.start && currentTime <= word.end);

    if (currentIndex !== -1) {
        // Only update if we reach the last word of the current block
        if (currentIndex >= currentBlockStart + captionWordLimit) {
            currentBlockStart = currentIndex; // Move to the next block
            lastCaptionUpdateTime = currentTime; // Update last update time

            // Store styles for this block only once
            blockWordStyles = captionsData.slice(currentBlockStart, Math.min(currentBlockStart + captionWordLimit, captionsData.length)).map(wordObj => ({
                textColor: textColors[Math.floor(Math.random() * textColors.length)],
                bgColor: bgColors[Math.floor(Math.random() * bgColors.length)],
                fontSize: fontSizes[Math.floor(Math.random() * fontSizes.length)],
                angle: angles[Math.floor(Math.random() * angles.length)]
            }));            
        }

        let endIdx = Math.min(currentBlockStart + captionWordLimit, captionsData.length);
        let currentBlockWords = captionsData.slice(currentBlockStart, endIdx);
        let displayedWords = "";

        if (selectedStyle !== "block-style") {
            displayedWords = currentBlockWords.map((wordObj) => {
                return (currentTime >= wordObj.start && currentTime <= wordObj.end)
                    ? `<span class="current-word">${wordObj.word}</span>` // Highlight spoken word
                    : wordObj.word;
            });
        } else {
            displayedWords = currentBlockWords.map((wordObj, index) => {
                let span = document.createElement("span");
                span.innerText = wordObj.word;
        
                // Retrieve previously stored styles for this block
                let style = blockWordStyles[index] || {};

                span.style.color = style.textColor || "#FFF";
                span.style.backgroundColor = style.bgColor || "#000";
                span.style.fontSize = style.fontSize || "1em";
                //span.classList.add(style.angle || "angle1"); // Apply stored angle
                span.classList.add("word-box"); // Applies bold block style

                // 🔹 Highlight spoken word
                if (currentTime >= wordObj.start && currentTime <= wordObj.end) {
                    span.classList.add("current-word");
                    span.classList.add(style.angle || "angle1");
                }

                return span.outerHTML;
            });
        }

        const newCaption = displayedWords.join(" ");

        if (newCaption !== captions.innerHTML) {
            captions.innerHTML = newCaption;
            captions.classList.add("show-caption");
            captions.classList.remove("hide-caption");
        }
    } else if (currentTime - lastCaptionUpdateTime < 2) {
        // 🔹 If there’s a pause, keep the last caption visible for 2 seconds
        captions.classList.add("show-caption");
        captions.classList.remove("hide-caption");
    } else if (captions.innerHTML !== "") {
        // 🔹 After the pause, fade out the caption
        captions.classList.add("hide-caption");
        setTimeout(() => captions.classList.remove("show-caption"), 300);
    }
});


// 🔹 Simulate Caption Animation in Preview Section
function startPreviewAnimation() {
    clearInterval(previewInterval); // Reset animation if already running
    previewTime = 0.0;

    previewInterval = setInterval(() => {
        previewTime += 0.1; // Move forward in time

        let currentIndex = dummyCaptionsData.findIndex(word => previewTime >= word.start && previewTime <= word.end);

        if (currentIndex !== -1) {
            let displayedWords = dummyCaptionsData.map((wordObj, index) => {
                return index === currentIndex
                    ? `<span class="current-word">${wordObj.word}</span>` // Highlight current word
                    : wordObj.word;
            });

            captionPreview.innerHTML = displayedWords.join(" ");
        }

        if (previewTime >= 2.0) {
            previewTime = 0.0; // Restart the animation loop
        }
    }, 100);
}

// Start the caption animation loop on page load
startPreviewAnimation();

// 🔹 Render Editable Words
function renderWordEditor() {
    wordEditor.innerHTML = ""; // Clear previous content

    wordTimestamps.forEach((wordObj, index) => {
        let wordDiv = document.createElement("div");
        wordDiv.classList.add("word-editor-box");

        // Editable input field
        let input = document.createElement("input");
        input.type = "text";
        input.value = wordObj.word;
        input.dataset.index = index;

        // Delete button
        let deleteBtn = document.createElement("span");
        deleteBtn.innerHTML = "❌";
        deleteBtn.classList.add("delete-word");
        deleteBtn.dataset.index = index;

        // View Details Button
        let detailsBtn = document.createElement("span");
        detailsBtn.innerHTML = "ℹ️";  // Info icon
        detailsBtn.classList.add("view-details");
        detailsBtn.dataset.index = index;

        wordDiv.appendChild(input);
        wordDiv.appendChild(detailsBtn); // Add details button
        wordDiv.appendChild(deleteBtn);
        wordEditor.appendChild(wordDiv);
    });
}

// 🔹 Handle Editing
wordEditor.addEventListener("input", (event) => {
    if (event.target.tagName === "INPUT") {
        let index = event.target.dataset.index;
        wordTimestamps[index].word = event.target.value.trim();
    }
});

// Function to create the floating tooltip for word properties
function showWordDetails(index, event) {
    let wordObj = wordTimestamps[index];

    // Remove existing tooltip
    let existingTooltip = document.querySelector(".word-tooltip");
    if (existingTooltip) existingTooltip.remove();

    // Create tooltip div
    let tooltip = document.createElement("div");
    tooltip.classList.add("word-tooltip");
    tooltip.innerHTML = `
        <strong>📖 Word:</strong> ${wordObj.word} <br>
        <strong>⏳ Start:</strong> <span class="copy-text">${wordObj.start}</span> <button class="copy-btn" data-text="${wordObj.start}">📋</button><br>
        <strong>⏳ End:</strong> <span class="copy-text">${wordObj.end}</span> <button class="copy-btn" data-text="${wordObj.end}">📋</button>
    `;

    // Position tooltip near the clicked button
    tooltip.style.position = "absolute";
    tooltip.style.left = `${event.clientX + 10}px`;
    tooltip.style.top = `${event.clientY + 10}px`;
    
    document.body.appendChild(tooltip);
}

// Event listener for copying times to clipboard
document.body.addEventListener("click", (event) => {
    if (event.target.classList.contains("copy-btn")) {
        let textToCopy = event.target.dataset.text;
        navigator.clipboard.writeText(textToCopy).then(() => {
            event.target.innerText = "✅";  // Show checkmark after copying
            setTimeout(() => event.target.innerText = "📋", 1000);
        });
    }
});

// Event listener for showing details
wordEditor.addEventListener("click", (event) => {
    if (event.target.classList.contains("view-details")) {
        let index = event.target.dataset.index;
        showWordDetails(index, event);
    }
});

// 🔹 Handle Deletion
wordEditor.addEventListener("click", (event) => {
    if (event.target.classList.contains("delete-word")) {
        let index = event.target.dataset.index;
        wordTimestamps.splice(index, 1);
        renderWordEditor(); // Re-render after deletion
    }
});

// 🔹 Save Updated Words to `word_timestamps.json`
// 🔹 Save Updated Words to `word_timestamps.json` via Python server
saveWordChanges.addEventListener("click", () => {
    fetch("http://localhost:5000/save_word_timestamps", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(wordTimestamps)
    })
    .then(response => response.json())
    .then(data => alert(data.message))
    .catch(error => console.error("Error saving word timestamps:", error));
});