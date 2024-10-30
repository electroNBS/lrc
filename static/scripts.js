document.getElementById("captureBtn").addEventListener("click", () => {
  fetch("/capture", { method: "POST" })
    .then((response) => response.json())
    .then((data) => {
      console.log(data);
      document.getElementById("resultImage").src = data.image_url;
    });
});

document.getElementById("previewBtn").addEventListener("click", () => {
  const offset = document.getElementById("offset").value;
  const tolerance = document.getElementById("tolerance").value;
  fetch("/process", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      offset: parseInt(offset),
      tolerance: parseInt(tolerance),
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      document.getElementById("resultImage").src = data.image_url;
    });
});

document.getElementById("exportBtn").addEventListener("click", () => {
  window.location.href = "/export";

});
