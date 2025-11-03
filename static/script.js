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
    charCount.dataset.state = length > MAX_PROMPT_LENGTH ? "error" : "ok";
}

function setLoading(isLoading) {
    if (!recommendBtn) {
        return;
    }

    recommendBtn.disabled = isLoading;
    randomBtn.disabled = isLoading;
    promptInput.setAttribute("aria-busy", String(isLoading));

    if (isLoading) {
        feedback.textContent = "Mixing tracks…";
        feedback.classList.remove("error");
        recommendBtn.classList.add("is-loading");
        statusPill.textContent = "Fetching";
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
        </div>
    `;
    recommendationsGrid.classList.add("empty");
    showSuccess("");
}

function surprisePrompt() {
    const randomPrompt = presetPrompts[Math.floor(Math.random() * presetPrompts.length)];
    promptInput.value = randomPrompt;
    updateCharCount();
}

function renderRecommendations(payload) {
    recommendationsGrid.innerHTML = "";

    if (!payload || !Array.isArray(payload.recommendations) || payload.recommendations.length === 0) {
        clearRecommendations();
        return;
    }

    recommendationsGrid.classList.remove("empty");

    payload.recommendations.forEach((item, index) => {
        const card = document.createElement("article");
        card.className = "recommendation-card";
        card.innerHTML = `
            <h3>${item.title || `Track ${index + 1}`}</h3>
            <p class="recommendation-meta">${item.artist || "Unknown artist"} · ${item.genre || "Unclassified"}</p>
            <p class="recommendation-description">${item.description || item.reason || "No description provided."}</p>
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
            body: JSON.stringify({ prompt }),
            signal: currentRequestController.signal
        });

        if (response.status === 429) {
            const retryAfter = response.headers.get("Retry-After");
            showError(
                retryAfter
                    ? `Traffic is busy. Try again in ${retryAfter} seconds.`
                    : "Traffic is busy. Try again shortly."
            );
            return;
        }

        if (!response.ok) {
            throw new Error(`Server responded with status ${response.status}`);
        }

        const data = await response.json();
        renderRecommendations(data);
        showSuccess("Here are some tracks to explore.");
    } catch (error) {
        if (error.name === "AbortError") {
            return;
        }

        console.error(error);
        showError("We hit a sour note. Please try again.");
    } finally {
        setLoading(false);
        currentRequestController = null;
    }
}

promptForm?.addEventListener("submit", (event) => {
    event.preventDefault();
    const prompt = promptInput.value.trim();

    if (!prompt) {
        showError("Drop in a mood, genre mashup, or story first.");
        return;
    }

    if (prompt.length > MAX_PROMPT_LENGTH) {
        showError("That prompt is a little long. Try shortening it.");
        return;
    }

    fetchRecommendations(prompt);
});

randomBtn?.addEventListener("click", () => {
    surprisePrompt();
    promptInput.focus({ preventScroll: true });
});

resetBtn?.addEventListener("click", () => {
    promptInput.value = "";
    updateCharCount();
    clearRecommendations();
    promptInput.focus({ preventScroll: true });
});

promptInput?.addEventListener("input", updateCharCount);

promptInput?.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
        event.preventDefault();
        promptForm.requestSubmit();
    }
});

surprisePrompt();
updateCharCount();
