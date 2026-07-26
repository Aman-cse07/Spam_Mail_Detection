let currentMessage = "";

function predictSpam() {

    const message = document.getElementById("message").value;

    currentMessage = message;

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: `message=${encodeURIComponent(message)}`
    })
    .then(response => response.json())
    .then(data => {

        const resultDiv = document.getElementById("result");

        if(data.prediction === 'spam') {
            resultDiv.innerHTML = `🚨 SPAM MAIL<br>Confidence: ${data.confidence}`;
            resultDiv.className = 'spam-text';
        }
        else {
            resultDiv.innerHTML = `✅ SAFE MAIL<br>Confidence: ${data.confidence}`;
            resultDiv.className = 'ham-text';
        }

        document.getElementById("feedbackSection").style.display = 'block';
    });
}

function sendFeedback(label) {

    fetch('/feedback', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: `message=${encodeURIComponent(currentMessage)}&correct_label=${label}`
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
    });
}