// ------------------ Pagination ------------------
const rowsPerPage = 10;
let currentPage = 1;
let table, rows;

window.onload = function () {
  table = document.querySelector("table");
  if (!table) return;
  rows = table.querySelectorAll("tbody tr");
  showPage(currentPage);
};

function showPage(page) {
  const totalPages = Math.ceil(rows.length / rowsPerPage);
  if (page < 1 || page > totalPages) return;

  currentPage = page;
  const start = (page - 1) * rowsPerPage;
  const end = start + rowsPerPage;

  rows.forEach((row, index) => {
    row.style.display = (index >= start && index < end) ? "" : "none";
  });

  updatePaginationControls(totalPages);
}

function updatePaginationControls(totalPages) {
  const pagination = document.getElementById("pagination-controls");
  pagination.innerHTML = '';

  const prevBtn = document.createElement("button");
  prevBtn.textContent = "Précédent";
  prevBtn.disabled = (currentPage === 1);
  prevBtn.onclick = () => showPage(currentPage - 1);
  pagination.appendChild(prevBtn);

  const range = getPaginationRange(currentPage, totalPages);
  range.forEach(p => {
    if (p === '...') {
      const span = document.createElement("span");
      span.className = "ellipsis";
      span.textContent = "...";
      pagination.appendChild(span);
    } else {
      const btn = document.createElement("button");
      btn.textContent = p;
      if (p === currentPage) btn.classList.add("active");
      btn.onclick = () => showPage(p);
      pagination.appendChild(btn);
    }
  });

  const nextBtn = document.createElement("button");
  nextBtn.textContent = "Suivant";
  nextBtn.disabled = (currentPage === totalPages);
  nextBtn.onclick = () => showPage(currentPage + 1);
  pagination.appendChild(nextBtn);
}

function getPaginationRange(current, total) {
  const delta = 1;
  const range = [];
  const left = current - delta;
  const right = current + delta;
  let l;
  for (let i = 1; i <= total; i++) {
    if (i === 1 || i === total || (i >= left && i <= right)) {
      if (l && i - l !== 1) {
        range.push('...');
      }
      range.push(i);
      l = i;
    }
  }
  return range;
}

// ------------------ IA Explications ------------------
document.querySelectorAll('.btn-ai-explain').forEach(button => {
  button.addEventListener('click', function () {
    const targetId = this.getAttribute('data-target-id');
    const el = document.getElementById(targetId);

    const variable = el.dataset.variable;
    const lignes = el.dataset.lignes;
    const premiers = el.dataset.premiers;

    const prompt = `
    Voici l'ensemble de données :
    - nombre des lignes : ${lignes}
    - nombre des colonnes : ${variable}
    - 5 premiers lignes : ${premiers}

    Expliquez en détail cet ensemble de données en mentionnant :

    - Les noms et types des colonnes (sans donner les numéros des colonnes).
    - Toute observation pertinente (valeurs manquantes, tendances, etc.).

    Réponse : en français.`;

    let content = '';
    let finalText = '';
    let liveSeconds = 0;
    const startTime = performance.now();

    el.textContent = '⏱️ Temps réel : 0.0 sec';

    const timerId = setInterval(() => {
      el.textContent = content + `\n\n⏱️ Temps réel : ${liveSeconds.toFixed(1)} sec`;
      liveSeconds += 0.1;
    }, 100);

    fetch(`/stream_explanation/?prompt=${encodeURIComponent(prompt)}&id=${variable}`)
      .then(response => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        function read() {
          reader.read().then(({ done, value }) => {
            if (done) {
              clearInterval(timerId);
              const endTime = performance.now();
              const duration = ((endTime - startTime) / 1000).toFixed(2);
              el.textContent = finalText.trim() + `\n\n⏱️ Temps de génération : ${duration} sec`;
              return;
            }

            const chunk = decoder.decode(value, { stream: true });
            const words = chunk.split(/(\s+)/);
            let i = 0;

            function showWordByWord() {
              if (i < words.length) {
                content += words[i];
                finalText += words[i];
                i++;
                el.textContent = content + `\n\n⏱️ Temps réel : ${liveSeconds.toFixed(1)} sec`;
                setTimeout(showWordByWord, 40);
              } else {
                read();
              }
            }

            showWordByWord();
          });
        }

        read();
      })
      .catch(err => {
        clearInterval(timerId);
        el.textContent = "[Erreur de chargement]";
      });
  });
});



