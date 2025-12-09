const promptForm = document.getElementById("prompt-form");
const promptInput = document.getElementById("prompt-input");
const recommendBtn = document.getElementById("recommend-btn");
const randomBtn = document.getElementById("random-btn");
const resetBtn = document.getElementById("reset-btn");
const feedback = document.getElementById("feedback");
const charCount = document.getElementById("char-count");
const recommendationsGrid = document.getElementById("recommendations");
const statusPill = document.querySelector(".status-pill");

const MAX_PROMPT_LENGTH = 280;

// Demo preset prompts
const presetPrompts = [
    "Introspective hip-hop with dusty drums and string quartets",
    "Cinematic ambient score with analog synth blooms",
    "Feel-good funk for a sunny patio brunch",
    "Hyperpop energy with glitchy vocal chops and big drops",
    "Dreamy folk electronica with layered harmonies"
];

let currentRequestController = null;

function updateCharCount() {
    const length = promptInput.value.length;
    charCount.textContent = `${length} / ${MAX_PROMPT_LENGTH}`;
}

function setLoading(isLoading) {
    recommendBtn.disabled = isLoading;
    randomBtn.disabled = isLoading;

    if (isLoading) {
        feedback.textContent = "Mixing tracks…";
        statusPill.textContent = "Fetching";
        recommendBtn.classList.add("is-loading");
    } else {
        recommendBtn.classList.remove("is-loading");
        statusPill.textContent = "Ready";
    }
}

function showError(message) {
    feedback.textContent = message;
    feedback.classList.add("error");
}

function showSuccess(message) {
    feedback.textContent = message;
    feedback.classList.remove("error");
}

function clearRecommendations() {
    recommendationsGrid.innerHTML = `
        <div class="placeholder" aria-hidden="true">
            <span class="placeholder-icon">✨</span>
            <p>${recommendationsGrid.dataset.emptyLabel}</p>
        </div>`;
    recommendationsGrid.classList.add("empty");
}

function surprisePrompt() {
    const item = presetPrompts[Math.floor(Math.random() * presetPrompts.length)];
    promptInput.value = item;
    updateCharCount();
}

function renderRecommendations(payload) {
    recommendationsGrid.innerHTML = "";
    recommendationsGrid.classList.remove("empty");

    (payload.recommendations || []).forEach((track) => {
        const card = document.createElement("article");
        card.className = "recommendation-card";

        card.innerHTML = `
            <h3>${track.title ?? "Unknown title"}</h3>
            <p class="recommendation-meta">${track.artist ?? "Unknown artist"} · ${track.release_year ?? ""}</p>
            <p class="recommendation-description">Score: ${track.score?.toFixed(3) ?? "N/A"}</p>
        `;

        recommendationsGrid.appendChild(card);
    });
}

async function fetchRecommendations(prompt) {
    if (currentRequestController) {
        currentRequestController.abort();
    }

    currentRequestController = new AbortController();

    try {
        setLoading(true);

        const response = await fetch("/api/recommend", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                prompt,
                user_id: "demo_user" // REQUIRED FIX
            }),
            signal: currentRequestController.signal
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        renderRecommendations(data);
        showSuccess("Here are your tracks!");

    } catch (err) {
        if (err.name !== "AbortError") {
            console.error(err);
            showError("Something went wrong. Try again.");
        }
    } finally {
        setLoading(false);
        currentRequestController = null;
    }
}

promptForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = promptInput.value.trim();
    if (!text) {
        showError("Tell me a vibe first.");
        return;
    }
    fetchRecommendations(text);
});

randomBtn.addEventListener("click", () => {
    surprisePrompt();
});

resetBtn.addEventListener("click", () => {
    promptInput.value = "";
    updateCharCount();
    clearRecommendations();
});

promptInput.addEventListener("input", updateCharCount);

surprisePrompt();
updateCharCount();
