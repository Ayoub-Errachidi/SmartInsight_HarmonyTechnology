export function formatAIChunk(textChunk, state) {
  let displayText = '';
  const chars = [...textChunk];

  for (const char of chars) {
    // Gestion des astérisques **
    if (char === '*') {
      state.asteriskCount++;

      if (state.asteriskCount === 2) {
        state.isBold = !state.isBold;

        if (!state.isBold) {
          displayText += `<strong>${state.boldBuffer}</strong>`;
          state.boldBuffer = '';
        }
        continue;
      }
      continue;
    } else {
      state.asteriskCount = 0;
    }

    // Si on est dans une séquence en gras
    if (state.isBold) {
      state.boldBuffer += char;
    } else {
      displayText += (char === '\n') ? '<br>' : char;
    }
  }

  return displayText;
}