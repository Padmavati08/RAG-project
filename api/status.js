const { documents, chunks } = require('./utils');

module.exports = (req, res) => {
  res.setHeader('Content-Type', 'application/json');
  res.status(200).json({
    ready: true,
    totalDocuments: documents.length,
    totalChunks: chunks.length,
    activeModel: 'Local Hybrid RAG Engine (no external key)',
    documents: documents.map(d => ({ id: d.id, title: d.title, numPages: d.numPages, fileName: d.fileName, uploadedAt: d.uploadedAt }))
  });
};
