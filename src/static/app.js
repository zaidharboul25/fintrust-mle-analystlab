/**
 * FinTrust AI Guard - Frontend Application Logic
 * Vanilla JavaScript (No Framework Dependency)
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const healthBadge = document.getElementById("systemHealthBadge");
    const healthText = document.getElementById("healthText");
    const predictionForm = document.getElementById("predictionForm");
    const submitBtn = document.getElementById("submitBtn");
    const submitBtnText = document.getElementById("submitBtnText");
    const latencyBadge = document.getElementById("latencyBadge");

    // Result Card Elements
    const resultCard = document.getElementById("resultCard");
    const emptyState = document.getElementById("emptyState");
    const activeResult = document.getElementById("activeResult");
    const decisionBadge = document.getElementById("decisionBadge");
    const decisionExplanation = document.getElementById("decisionExplanation");
    const confidencePercent = document.getElementById("confidencePercent");
    const meterFill = document.getElementById("meterFill");
    const auditTxId = document.getElementById("auditTxId");
    const auditModel = document.getElementById("auditModel");
    const auditLatency = document.getElementById("auditLatency");
    const auditTimestamp = document.getElementById("auditTimestamp");

    // Inspectors
    const jsonViewer = document.getElementById("jsonViewer");
    const curlViewer = document.getElementById("curlViewer");
    const historyBody = document.getElementById("historyBody");
    const clearHistoryBtn = document.getElementById("clearHistoryBtn");

    // Presets Buttons
    const presetSafeBtn = document.getElementById("presetSafeBtn");
    const presetRiskBtn = document.getElementById("presetRiskBtn");
    const presetNewCustomerBtn = document.getElementById("presetNewCustomerBtn");
    const resetFormBtn = document.getElementById("resetFormBtn");

    let historyRecords = [];

    // =====================================================================
    // 1. Initial Health Check
    // =====================================================================
    async function checkSystemHealth() {
        try {
            const response = await fetch("/health");
            if (response.ok) {
                const data = await response.json();
                healthBadge.className = "health-pill healthy";
                healthText.textContent = `En ligne (${data.model_version || "ML v1"}) • ${data.features_count || 68} features`;
            } else {
                healthBadge.className = "health-pill degraded";
                healthText.textContent = "Service dégradé";
            }
        } catch (error) {
            console.error("Health check error:", error);
            healthBadge.className = "health-pill degraded";
            healthText.textContent = "API non joignable";
        }
    }

    checkSystemHealth();
    setInterval(checkSystemHealth, 30000); // Check every 30s

    // =====================================================================
    // 2. Form Presets Management
    // =====================================================================
    const presets = {
        safe: {
            Transaction_ID: "FT-T009841",
            Customer_ID: "FT-C00124",
            Amount_NGN: 25000.00,
            Transaction_Type: "Transfer",
            Channel: "Mobile App",
            Location: "Lagos",
            Device_Type: "Android",
            International_Transaction: "No",
            Transaction_Status: "Successful",
            Transaction_DateTime: "2026-10-08 14:30:00",
            Customer_Name: "Amara Okafor",
            Age: 32,
            Gender: "Female",
            City: "Lagos",
            Customer_Segment: "Everyday",
            Account_Type: "Savings",
            Tenure_Months: 24,
            Monthly_Income_Band: "100k-249k",
            Digital_Engagement_Score: 4.2
        },
        risk: {
            Transaction_ID: "FT-T009999",
            Customer_ID: "FT-C00888",
            Amount_NGN: 890000.00,
            Transaction_Type: "Transfer",
            Channel: "Web",
            Location: "Abuja",
            Device_Type: "Unknown",
            International_Transaction: "Yes",
            Transaction_Status: "Successful",
            Transaction_DateTime: "2026-10-08 02:45:00",
            Customer_Name: "Tunde Bakare",
            Age: 22,
            Gender: "Male",
            City: "Lagos",
            Customer_Segment: "Student",
            Account_Type: "Savings",
            Tenure_Months: 2,
            Monthly_Income_Band: "Below 100k",
            Digital_Engagement_Score: 1.2
        },
        newCustomer: {
            Transaction_ID: "FT-T005512",
            Customer_ID: "FT-C00912",
            Amount_NGN: 150000.00,
            Transaction_Type: "Payment",
            Channel: "POS",
            Location: "Port Harcourt",
            Device_Type: "POS Terminal",
            International_Transaction: "Yes",
            Transaction_Status: "Successful",
            Transaction_DateTime: "2026-10-08 18:15:00",
            Customer_Name: "Fatima Danjuma",
            Age: 45,
            Gender: "Female",
            City: "Kano",
            Customer_Segment: "SME",
            Account_Type: "Current",
            Tenure_Months: 1,
            Monthly_Income_Band: "500k-999k",
            Digital_Engagement_Score: 2.5
        }
    };

    function applyPreset(presetData) {
        Object.keys(presetData).forEach(key => {
            const input = predictionForm.elements[key];
            if (input) {
                input.value = presetData[key];
                if (key === "Digital_Engagement_Score") {
                    document.getElementById("engagementValue").textContent = presetData[key];
                }
            }
        });
    }

    presetSafeBtn.addEventListener("click", () => applyPreset(presets.safe));
    presetRiskBtn.addEventListener("click", () => applyPreset(presets.risk));
    presetNewCustomerBtn.addEventListener("click", () => applyPreset(presets.newCustomer));
    resetFormBtn.addEventListener("click", () => {
        predictionForm.reset();
        document.getElementById("engagementValue").textContent = "3.8";
    });

    // =====================================================================
    // 3. Form Submission & Scoring Execution
    // =====================================================================
    predictionForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        await executePrediction();
    });

    // Keyboard shortcut: Ctrl + Enter / Cmd + Enter
    document.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            executePrediction();
        }
    });

    async function executePrediction() {
        const formData = new FormData(predictionForm);

        // Build Payload matching FastAPI PredictRequest schema
        const payload = {
            transaction: {
                Transaction_ID: formData.get("Transaction_ID") || "FT-T000001",
                Customer_ID: formData.get("Customer_ID") || "FT-C00001",
                Transaction_DateTime: formData.get("Transaction_DateTime") || new Date().toISOString().replace("T", " ").substring(0, 19),
                Transaction_Type: formData.get("Transaction_Type"),
                Amount_NGN: parseFloat(formData.get("Amount_NGN")),
                Channel: formData.get("Channel"),
                Device_Type: formData.get("Device_Type") || "Unknown",
                Location: formData.get("Location"),
                International_Transaction: formData.get("International_Transaction"),
                Transaction_Status: formData.get("Transaction_Status")
            },
            customer: {
                Customer_ID: formData.get("Customer_ID") || "FT-C00001",
                Customer_Name: formData.get("Customer_Name") || "Customer",
                Age: parseInt(formData.get("Age"), 10) || 30,
                Gender: formData.get("Gender"),
                City: formData.get("City"),
                Customer_Segment: formData.get("Customer_Segment"),
                Account_Type: formData.get("Account_Type"),
                Tenure_Months: parseInt(formData.get("Tenure_Months"), 10) || 12,
                Digital_Engagement_Score: parseFloat(formData.get("Digital_Engagement_Score")) || 3.0,
                Monthly_Income_Band: formData.get("Monthly_Income_Band"),
                Preferred_Channel: formData.get("Channel") === "POS" || formData.get("Channel") === "ATM" ? "Mobile App" : formData.get("Channel"),
                Account_Status: "Active"
            }
        };

        // Update cURL Inspector
        updateCurlViewer(payload);

        // Set Loading UI state
        submitBtn.disabled = true;
        submitBtnText.textContent = "Évaluation en cours...";
        latencyBadge.textContent = "Calcul...";

        const startTime = performance.now();

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                body: JSON.stringify(payload)
            });

            const durationMs = Math.round(performance.now() - startTime);

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "Erreur de prédiction");
            }

            const data = await response.json();
            renderPredictionResult(data, durationMs, payload);

        } catch (error) {
            console.error("Prediction error:", error);
            alert(`Erreur : ${error.message}`);
            latencyBadge.textContent = "Erreur";
        } finally {
            submitBtn.disabled = false;
            submitBtnText.textContent = "Évaluer le Risque de la Transaction";
        }
    }

    // =====================================================================
    // 4. Render Prediction Results
    // =====================================================================
    function renderPredictionResult(data, latencyMs, payload) {
        // Toggle view from empty state to active
        emptyState.style.display = "none";
        activeResult.style.display = "block";

        const isRisk = data.prediction === "Yes";
        const confidenceVal = Math.round(data.confidence * 100);

        // Update Badge
        if (isRisk) {
            decisionBadge.className = "decision-badge risk-review";
            decisionBadge.innerHTML = "⚠️ RISK REVIEW REQUIRED (ALERTE)";
            decisionExplanation.textContent = "Cette transaction dépasse le seuil critique de suspicion et nécessite une révision manuelle immédiate.";
            meterFill.className = "meter-fill risk";
        } else {
            decisionBadge.className = "decision-badge approved";
            decisionBadge.innerHTML = "✓ TRANSACTION NORMALE (CONFORME)";
            decisionExplanation.textContent = "Le profil de transaction est conforme aux comportements habituels du compte.";
            meterFill.className = "meter-fill safe";
        }

        // Animate meter
        confidencePercent.textContent = `${confidenceVal}%`;
        meterFill.style.width = `${confidenceVal}%`;

        // Update Audit Badges
        auditTxId.textContent = data.transaction_id;
        auditModel.textContent = data.model_version;
        auditLatency.textContent = `${latencyMs} ms`;
        auditTimestamp.textContent = new Date(data.prediction_timestamp).toLocaleTimeString();
        latencyBadge.textContent = `${latencyMs} ms`;

        // Update JSON viewer
        jsonViewer.textContent = JSON.stringify(data, null, 2);

        // Add to Session History
        addToHistory(data, payload);
    }

    function updateCurlViewer(payload) {
        const origin = window.location.origin || "http://localhost:8000";
        const curlCmd = `curl -X POST "${origin}/predict" \\\n  -H "Content-Type: application/json" \\\n  -d '${JSON.stringify(payload, null, 2)}'`;
        curlViewer.textContent = curlCmd;
    }

    function addToHistory(data, payload) {
        historyRecords.unshift({
            txId: data.transaction_id,
            amount: payload.transaction.Amount_NGN,
            decision: data.prediction,
            confidence: Math.round(data.confidence * 100),
            time: new Date().toLocaleTimeString()
        });

        if (historyRecords.length > 8) {
            historyRecords.pop();
        }

        renderHistoryTable();
    }

    function renderHistoryTable() {
        if (historyRecords.length === 0) {
            historyBody.innerHTML = `<tr><td colspan="5" class="empty-row">Aucune prédiction enregistrée pour cette session.</td></tr>`;
            return;
        }

        historyBody.innerHTML = historyRecords.map(item => `
            <tr>
                <td class="mono">${item.txId}</td>
                <td>₦ ${Number(item.amount).toLocaleString()}</td>
                <td>
                    <span class="mini-badge ${item.decision.toLowerCase()}">
                        ${item.decision === "Yes" ? "Alerte" : "Normal"}
                    </span>
                </td>
                <td class="mono">${item.confidence}%</td>
                <td>${item.time}</td>
            </tr>
        `).join("");
    }

    clearHistoryBtn.addEventListener("click", () => {
        historyRecords = [];
        renderHistoryTable();
    });

    // =====================================================================
    // 5. Inspector Tabs
    // =====================================================================
    const tabButtons = document.querySelectorAll(".tab-btn");
    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            tabButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            const tabId = btn.getAttribute("data-tab");
            document.querySelectorAll(".tab-content").forEach(content => {
                content.style.display = content.id === tabId ? "block" : "none";
            });
        });
    });
});
