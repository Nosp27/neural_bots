document.addEventListener('keydown', e => {
    if (e.key === 'Enter')
        onSend();
});

function onSend() {
    const message_element = document.getElementById('message');
    const your_message = message_element.value;
    if (your_message === '')
        return;
    const model_name = document.getElementById('model_name').innerText;
    message_element.value = '';
    fetch(`/answer?message=${your_message}&model_name=${model_name}`)
        .then(x => onResponse(your_message, x))
        .catch(console.log);
}

async function onResponse(your_message, x) {
    const jsonResp = await x.json();
    // alert(JSON.stringify(jsonResp));
    const messagesHtml = document.getElementById('messages');
    messagesHtml.innerHTML = (
        messagesHtml.innerHTML +
        `<p>${your_message}` +
        `<p>- ${jsonResp['message']}</p>`
    );
}