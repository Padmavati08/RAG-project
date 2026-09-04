const { chunks, retrieveTopK, localAnswerSynthesizer, evaluateResponseQuality } = require('./utils');

module.exports = async (req, res) => {
  try {
    if (req.method !== 'POST') return res.status(405).json({ error: 'POST required' });
    const body = req.body || {};
    const { query, topK = 5, documentId } = body;
    if (!query || typeof query !== 'string' || !query.trim()) return res.status(400).json({ error: 'query required' });

    const retrieved = retrieveTopK(query.trim(), chunks, Number(topK) || 5, documentId);
    const matchedChunks = retrieved.map(r => r.chunk);
    const answer = localAnswerSynthesizer(matchedChunks, query.trim());
    const evaluation = evaluateResponseQuality(query.trim(), answer, matchedChunks);

    return res.status(200).json({
      query: query.trim(),
      answer,
      context: matchedChunks.map((c,i) => `[Context ${i+1}]: ${c.text}`).join('\n\n'),
      evaluation,
      results: retrieved.map(r => ({ id: r.chunk.id, docId: r.chunk.docId, docTitle: r.chunk.docTitle, chunkIndex: r.chunk.chunkIndex, textSnippet: r.chunk.text.slice(0,200) + '...', score: Math.round(r.score*100)/100 })),
      isAiGenerated: false
    });
  } catch (err) {
    console.error('[api/query] error', err);
    return res.status(500).json({ error: err.message || 'internal error' });
  }
};
