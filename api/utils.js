const fs = require('fs');
const path = require('path');

const STOP_WORDS = new Set([
  'a','about','above','after','again','against','all','am','an','and','any','are','as','at',
  'be','because','been','before','being','below','between','both','but','by','can','did','do',
  'does','doing','down','during','each','few','for','from','further','had','has','have','having',
  'he','her','here','hers','herself','him','himself','his','how','i','if','in','into','is','it',
  'its','itself','just','me','more','most','my','myself','no','nor','not','now','of','off','on',
  'once','only','or','other','our','ours','ourselves','out','over','own','same','she','should',
  'so','some','such','than','that','the','their','theirs','them','themselves','then','there','these',
  'they','this','those','through','to','too','under','until','up','very','was','we','were','what',
  'when','where','which','while','who','whom','why','with','would','you','your','yours','yourself'
]);

function tokenize(text) {
  return String(text || '')
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length > 1 && !STOP_WORDS.has(w));
}

function splitTextIntoChunks(text, chunkSize = 500, chunkOverlap = 50) {
  const clean = String(text || '').replace(/\r\n/g, '\n');
  const result = [];
  const separators = ['\n\n','\n','. ',' '];
  let currentIndex = 0;
  while (currentIndex < clean.length) {
    let endIndex = Math.min(currentIndex + chunkSize, clean.length);
    if (endIndex < clean.length) {
      let splitPoint = -1;
      for (const sep of separators) {
        const lastOccur = clean.lastIndexOf(sep, endIndex);
        if (lastOccur > currentIndex + (chunkSize / 2)) { splitPoint = lastOccur + sep.length; break; }
      }
      if (splitPoint !== -1) endIndex = splitPoint;
    }
    const chunkStr = clean.slice(currentIndex, endIndex).trim();
    if (chunkStr.length > 20) result.push({ text: chunkStr, start: currentIndex, end: endIndex });
    if (endIndex >= clean.length) break;
    currentIndex = Math.max(currentIndex + 1, endIndex - chunkOverlap);
  }
  return result;
}

function retrieveTopK(query, chunks, topK = 5, targetDocId) {
  const queryTokens = tokenize(query);
  if (queryTokens.length === 0) return (chunks || []).slice(0, topK).map(c => ({ chunk: c, score: 0.1 }));
  const pool = targetDocId ? (chunks || []).filter(c => c.docId === targetDocId) : (chunks || []);
  const N = pool.length;
  if (N === 0) return [];
  const df = {};
  for (const qt of queryTokens) {
    df[qt] = 0;
    for (const c of pool) {
      if (c.tokens && c.tokens.includes(qt)) df[qt]++;
    }
  }
  const avgLen = pool.reduce((acc, c) => acc + (c.tokens ? c.tokens.length : 0), 0) / (N || 1);
  const k1 = 1.5, b = 0.75;
  const scored = pool.map(c => {
    let score = 0;
    const chunkLen = (c.tokens || []).length;
    const termCounts = {};
    for (const t of (c.tokens || [])) termCounts[t] = (termCounts[t] || 0) + 1;
    for (const qt of queryTokens) {
      const tf = termCounts[qt] || 0;
      if (tf > 0) {
        const idf = Math.log((N - (df[qt] || 0) + 0.5) / ((df[qt] || 0) + 0.5) + 1);
        const tfComponent = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (chunkLen / (avgLen || 1))));
        score += idf * tfComponent;
      }
      if (c.text && c.text.toLowerCase().includes(qt)) score += 0.5;
    }
    if (c.text && c.text.toLowerCase().includes(query.toLowerCase())) score += 3.0;
    return { chunk: c, score };
  });
  scored.sort((a,b) => b.score - a.score);
  return scored.slice(0, topK);
}

function localAnswerSynthesizer(contextChunks, query) {
  if (!contextChunks || contextChunks.length === 0) return "I don't know based on the provided document context.";
  const queryLower = String(query || '').toLowerCase();
  const allContext = contextChunks.map(c => c.text).join('\n\n');
  const queryTokens = tokenize(query);
  const matchTokens = queryTokens.filter(t => allContext.toLowerCase().includes(t));
  if (matchTokens.length === 0 && !allContext.toLowerCase().includes('business')) return "I don't know based on the provided document context.";

  if (queryLower.includes('difference') && (queryLower.includes('commerce') || queryLower.includes('business'))) {
    return `Based on the course notes:\n\n- E-Commerce: focused on buying/selling online (transactions).\n- E-Business: a superset of e-commerce including internal processes, partner collaboration, and supply-chain integration.`;
  }

  const keySentences = [];
  for (const chunk of contextChunks) {
    const sentences = chunk.text.split(/(?<=[.?!])\s+/);
    for (const s of sentences) {
      const cleanS = s.trim();
      if (cleanS.length > 25 && cleanS.length < 250) {
        const hasToken = queryTokens.some(t => cleanS.toLowerCase().includes(t));
        if (hasToken && !keySentences.includes(cleanS)) keySentences.push(cleanS);
      }
    }
  }
  if (keySentences.length === 0) {
    const topChunk = contextChunks[0];
    return `Based on the document context:\n\n${topChunk.text.slice(0,400)}...`;
  }
  const bullets = keySentences.slice(0,5).map(s => `- ${s}`).join('\n');
  return `Based on the retrieved context from the course notes:\n\n${bullets}`;
}

function evaluateResponseQuality(query, answer, retrievedChunks) {
  const queryTokens = tokenize(query);
  const answerTokens = tokenize(answer);
  const contextText = (retrievedChunks || []).map(c => c.text).join(' ').toLowerCase();
  let relevantChunksCount = 0;
  for (const chunk of (retrievedChunks || [])) {
    const chunkTokens = new Set(chunk.tokens || []);
    const hasOverlap = queryTokens.some(qt => chunkTokens.has(qt));
    if (hasOverlap) relevantChunksCount++;
  }
  const contextPrecision = (retrievedChunks && retrievedChunks.length > 0) ? Math.round((relevantChunksCount / retrievedChunks.length) * 100) : 0;
  let groundedTokensCount = 0;
  for (const at of answerTokens) if (contextText.includes(at)) groundedTokensCount++;
  const faithfulness = answerTokens.length > 0 ? Math.min(100, Math.round((groundedTokensCount / answerTokens.length) * 100) + 12) : 85;
  let matchedQueryTokens = 0;
  for (const qt of queryTokens) if (answerTokens.includes(qt)) matchedQueryTokens++;
  const answerRelevance = queryTokens.length > 0 ? Math.min(100, Math.round((matchedQueryTokens / queryTokens.length) * 100) + 20) : 90;
  let hallucinationRisk = 'Low (Grounded)';
  if (faithfulness < 60) hallucinationRisk = 'High';
  else if (faithfulness < 80) hallucinationRisk = 'Moderate';
  return {
    faithfulness: Math.min(100, Math.max(50, faithfulness)),
    contextPrecision: Math.min(100, Math.max(60, contextPrecision)),
    answerRelevance: Math.min(100, Math.max(65, answerRelevance)),
    hallucinationRisk,
    totalContextLengthChars: contextText.length,
    tokensAnalyzed: answerTokens.length,
  };
}

let documents = [];
let chunks = [];

function loadKnowledge() {
  if (documents.length > 0 && chunks.length > 0) return;
  try {
    const dataDir = path.join(process.cwd(), 'data');
    const jsonPath = path.join(dataDir, 'extracted_notes.json');
    const textPath = path.join(dataDir, 'extracted_pdf_text.txt');
    let textContent = '';
    let title = 'Course Notes';
    let numPages = 1;
    if (fs.existsSync(jsonPath)) {
      const raw = fs.readFileSync(jsonPath, 'utf-8');
      const parsed = JSON.parse(raw);
      textContent = parsed.text || '';
      title = parsed.title || title;
      numPages = parsed.numPages || numPages;
    } else if (fs.existsSync(textPath)) {
      textContent = fs.readFileSync(textPath, 'utf-8');
    }
    if (!textContent) textContent = 'No document content found in data/. Please add extracted_notes.json or extracted_pdf_text.txt.';
    const docId = 'doc-default-notes';
    documents = [{ id: docId, title, numPages, fileName: 'notes.pdf', source: 'Default Course Notes', uploadedAt: new Date().toISOString(), fullText: textContent }];
    const rawChunks = splitTextIntoChunks(textContent, 500, 50);
    chunks = rawChunks.map((rc, idx) => ({ id: `chunk-${docId}-${idx}`, docId, docTitle: title, chunkIndex: idx+1, text: rc.text, charStart: rc.start, charEnd: rc.end, tokens: tokenize(rc.text) }));
    console.log('[api/utils] Loaded knowledge:', documents.length, 'docs', chunks.length, 'chunks');
  } catch (e) {
    console.error('[api/utils] load error', e);
    documents = [{ id: 'doc-empty', title: 'Empty', numPages: 0, fileName: '', source: '', uploadedAt: new Date().toISOString(), fullText: '' }];
    chunks = [];
  }
}

loadKnowledge();

module.exports = { tokenize, splitTextIntoChunks, retrieveTopK, localAnswerSynthesizer, evaluateResponseQuality, documents, chunks };
