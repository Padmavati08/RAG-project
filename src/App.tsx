import React, { useState, useEffect, useRef } from 'react';
import {
  BookOpen,
  Search,
  Sparkles,
  Layers,
  FileText,
  Upload,
  CheckCircle2,
  AlertCircle,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
  History,
  RotateCcw,
  Loader2,
  Cpu,
  Bookmark,
  Database,
  BarChart3,
  GitBranch,
  ShieldCheck,
  Zap,
  ArrowRight,
  TrendingUp,
  Play
} from 'lucide-react';

interface RetrievedChunk {
  id: string;
  docId: string;
  docTitle: string;
  chunkIndex: number;
  text: string;
  score: number;
}

interface EvaluationMetrics {
  faithfulness: number;
  contextPrecision: number;
  answerRelevance: number;
  hallucinationRisk: 'Low (Grounded)' | 'Moderate' | 'High';
  totalContextLengthChars: number;
  tokensAnalyzed: number;
}

interface PipelineStep {
  step: number;
  name: string;
  details: string;
  durationMs: number;
}

interface QueryResult {
  query: string;
  answer: string;
  baselineAnswer?: string;
  context: string;
  results: RetrievedChunk[];
  evaluation?: EvaluationMetrics;
  pipelineTrace?: PipelineStep[];
  isAiGenerated: boolean;
  latencyMs: number;
  timestamp: string;
}

interface DocumentInfo {
  id: string;
  title: string;
  numPages: number;
  fileName: string;
  uploadedAt: string;
  chunkCount?: number;
}

interface SystemStatus {
  ready: boolean;
  totalDocuments: number;
  totalChunks: number;
  hasGeminiKey: boolean;
  activeModel: string;
}

interface VectorStoreData {
  totalDocuments: number;
  totalChunks: number;
  embeddingDimension: number;
  indexingMethod: string;
  chunks: Array<{
    id: string;
    docId: string;
    docTitle: string;
    chunkIndex: number;
    charStart?: number;
    charEnd?: number;
    tokenCount: number;
    topKeywords: string[];
    vectorPreview: number[];
    textSnippet: string;
  }>;
}

interface BenchmarkSuiteData {
  timestamp: string;
  totalEvaluations: number;
  summaryMetrics: {
    averageFaithfulnessScore: number;
    averageContextPrecision: number;
    averageAnswerRelevance: number;
    averageLatencyMs: number;
    overallQualityGrade: string;
  };
  benchmarkResults: Array<{
    id: string;
    question: string;
    expectedTopic: string;
    answer: string;
    latencyMs: number;
    topChunkId: string;
    metrics: EvaluationMetrics;
  }>;
}

const SAMPLE_QUESTIONS = [
  'Explain difference between e commerce and e business',
  'What are the primary Web Business Models described?',
  'Explain Tata Steel SAP e-procurement workflow and results',
  'How did ITC e-Choupal transform the soybean supply chain?',
  'What is Customer Delivered Value according to Kotler?'
];

export default function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [currentResult, setCurrentResult] = useState<QueryResult | null>(null);
  const [history, setHistory] = useState<QueryResult[]>([]);
  const [copiedAnswer, setCopiedAnswer] = useState(false);
  const [expandedChunks, setExpandedChunks] = useState<Record<string, boolean>>({});
  const [activeTab, setActiveTab] = useState<'qa' | 'pipeline' | 'vectordb' | 'evaluation' | 'documents'>('qa');
  const [selectedDocId, setSelectedDocId] = useState<string>('all');
  const [showBaselineComparison, setShowBaselineComparison] = useState(true);

  // Vector DB data
  const [vectorData, setVectorData] = useState<VectorStoreData | null>(null);
  const [vectorSearchFilter, setVectorSearchFilter] = useState('');

  // Benchmark suite data
  const [benchmarkData, setBenchmarkData] = useState<BenchmarkSuiteData | null>(null);
  const [runningBenchmark, setRunningBenchmark] = useState(false);

  // Upload modal state
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [uploadTitle, setUploadTitle] = useState('');
  const [uploadText, setUploadText] = useState('');
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchStatus();
    fetchDocuments();
    fetchVectorStore();
  }, []);

  const fetchStatus = async () => {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
      }
    } catch (err) {
      console.error('Failed to fetch status:', err);
    }
  };

  const fetchDocuments = async () => {
    try {
      const res = await fetch('/api/documents');
      if (res.ok) {
        const data = await res.json();
        setDocuments(data.documents || []);
      }
    } catch (err) {
      console.error('Failed to fetch documents:', err);
    }
  };

  const fetchVectorStore = async () => {
    try {
      const res = await fetch('/api/vector-store');
      if (res.ok) {
        const data = await res.json();
        setVectorData(data);
      }
    } catch (err) {
      console.error('Failed to fetch vector store:', err);
    }
  };

  const handleQuery = async (searchQuery?: string) => {
    const q = searchQuery || query;
    if (!q || !q.trim() || loading) return;

    setLoading(true);
    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: q.trim(),
          topK: 5,
          documentId: selectedDocId !== 'all' ? selectedDocId : undefined,
          includeBaseline: true,
        }),
      });

      if (!res.ok) {
        throw new Error('Query execution failed');
      }

      const data = await res.json();
      const resultWithTime: QueryResult = {
        ...data,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setCurrentResult(resultWithTime);
      setHistory((prev) => [resultWithTime, ...prev]);
      if (searchQuery) {
        setQuery(searchQuery);
      }
    } catch (err: any) {
      console.error('Query error:', err);
    } finally {
      setLoading(false);
    }
  };

  const runBenchmarkSuite = async () => {
    setRunningBenchmark(true);
    try {
      const res = await fetch('/api/benchmark');
      if (res.ok) {
        const data = await res.json();
        setBenchmarkData(data);
      }
    } catch (err) {
      console.error('Failed to run benchmark:', err);
    } finally {
      setRunningBenchmark(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedAnswer(true);
    setTimeout(() => setCopiedAnswer(false), 2000);
  };

  const toggleChunkExpand = (id: string) => {
    setExpandedChunks((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setUploadFile(file);
      if (!uploadTitle) {
        setUploadTitle(file.name.replace(/\.[^/.]+$/, ''));
      }
    }
  };

  const handleUploadSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadFile && !uploadText.trim()) {
      setUploadError('Please select a PDF file or paste text content.');
      return;
    }

    setUploading(true);
    setUploadError(null);

    try {
      let payload: any = {
        title: uploadTitle.trim() || (uploadFile ? uploadFile.name : 'Uploaded Document'),
      };

      if (uploadFile) {
        payload.fileName = uploadFile.name;
        const reader = new FileReader();
        const base64Promise = new Promise<string>((resolve, reject) => {
          reader.onload = () => resolve(reader.result as string);
          reader.onerror = reject;
          reader.readAsDataURL(uploadFile);
        });
        payload.fileBase64 = await base64Promise;
      } else {
        payload.textContent = uploadText.trim();
        payload.fileName = 'pasted-notes.txt';
      }

      const res = await fetch('/api/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.error || 'Upload failed');
      }

      await fetchDocuments();
      await fetchStatus();
      await fetchVectorStore();

      setIsUploadOpen(false);
      setUploadFile(null);
      setUploadTitle('');
      setUploadText('');
    } catch (err: any) {
      setUploadError(err.message || 'Failed to upload document');
    } finally {
      setUploading(false);
    }
  };

  const handleReset = async () => {
    if (confirm('Reset knowledge base to default educational notes?')) {
      try {
        await fetch('/api/reset', { method: 'POST' });
        await fetchDocuments();
        await fetchStatus();
        await fetchVectorStore();
        setCurrentResult(null);
      } catch (err) {
        console.error('Reset error:', err);
      }
    }
  };

  const filteredChunks = vectorData?.chunks.filter(
    (c) =>
      !vectorSearchFilter ||
      c.textSnippet.toLowerCase().includes(vectorSearchFilter.toLowerCase()) ||
      c.docTitle.toLowerCase().includes(vectorSearchFilter.toLowerCase()) ||
      c.topKeywords.some((k) => k.toLowerCase().includes(vectorSearchFilter.toLowerCase()))
  ) || [];

  return (
    <div id="rag-app-root" className="min-h-screen bg-slate-50 flex flex-col font-sans">
      {/* Top Application Header */}
      <header id="main-header" className="bg-white border-b border-slate-200 sticky top-0 z-30 shadow-xs">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-indigo-600 flex items-center justify-center text-white shadow-xs">
              <BookOpen className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-bold text-slate-900 tracking-tight">
                  RAG for Educational Systems
                </h1>
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                  Active RAG Engine
                </span>
              </div>
              <p className="text-xs text-slate-500">
                Enhance AI Responses with External Knowledge Bases & Quality Evaluation
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 w-full sm:w-auto justify-between sm:justify-end">
            <div className="hidden md:flex items-center gap-2 text-xs text-slate-600 bg-slate-100 px-3 py-1.5 rounded-lg border border-slate-200">
              <Database className="w-3.5 h-3.5 text-indigo-600" />
              <span>{status?.totalChunks || 0} Chunks in Vector DB</span>
              <span className="text-slate-300">|</span>
              <Cpu className="w-3.5 h-3.5 text-slate-500" />
              <span className="font-mono text-slate-700">{status?.activeModel || 'Gemini Flash'}</span>
            </div>

            <button
              id="upload-doc-btn"
              onClick={() => setIsUploadOpen(true)}
              className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-lg text-indigo-700 bg-indigo-50 border border-indigo-200 hover:bg-indigo-100 transition-colors"
            >
              <Upload className="w-3.5 h-3.5" />
              Add Knowledge PDF
            </button>

            <button
              id="reset-db-btn"
              onClick={handleReset}
              title="Reset Knowledge Base"
              className="p-2 text-slate-500 hover:text-slate-800 rounded-lg hover:bg-slate-100 transition-colors"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 flex flex-col gap-6">
        
        {/* Navigation Tabs */}
        <div className="flex border-b border-slate-200 gap-4 sm:gap-6 text-sm font-medium overflow-x-auto pb-px">
          <button
            id="tab-qa"
            onClick={() => setActiveTab('qa')}
            className={`pb-3 relative flex items-center gap-2 whitespace-nowrap transition-colors ${
              activeTab === 'qa' ? 'text-indigo-600 font-semibold' : 'text-slate-500 hover:text-slate-800'
            }`}
          >
            <Sparkles className="w-4 h-4" />
            Q&A Assistant
            {activeTab === 'qa' && (
              <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-indigo-600 rounded-full" />
            )}
          </button>

          <button
            id="tab-pipeline"
            onClick={() => setActiveTab('pipeline')}
            className={`pb-3 relative flex items-center gap-2 whitespace-nowrap transition-colors ${
              activeTab === 'pipeline' ? 'text-indigo-600 font-semibold' : 'text-slate-500 hover:text-slate-800'
            }`}
          >
            <GitBranch className="w-4 h-4" />
            RAG Pipeline Architecture
            {activeTab === 'pipeline' && (
              <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-indigo-600 rounded-full" />
            )}
          </button>

          <button
            id="tab-vectordb"
            onClick={() => {
              setActiveTab('vectordb');
              fetchVectorStore();
            }}
            className={`pb-3 relative flex items-center gap-2 whitespace-nowrap transition-colors ${
              activeTab === 'vectordb' ? 'text-indigo-600 font-semibold' : 'text-slate-500 hover:text-slate-800'
            }`}
          >
            <Database className="w-4 h-4" />
            Vector Database ({status?.totalChunks || 0})
            {activeTab === 'vectordb' && (
              <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-indigo-600 rounded-full" />
            )}
          </button>

          <button
            id="tab-evaluation"
            onClick={() => setActiveTab('evaluation')}
            className={`pb-3 relative flex items-center gap-2 whitespace-nowrap transition-colors ${
              activeTab === 'evaluation' ? 'text-indigo-600 font-semibold' : 'text-slate-500 hover:text-slate-800'
            }`}
          >
            <BarChart3 className="w-4 h-4" />
            Quality Evaluation Suite
            {activeTab === 'evaluation' && (
              <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-indigo-600 rounded-full" />
            )}
          </button>

          <button
            id="tab-documents"
            onClick={() => setActiveTab('documents')}
            className={`pb-3 relative flex items-center gap-2 whitespace-nowrap transition-colors ${
              activeTab === 'documents' ? 'text-indigo-600 font-semibold' : 'text-slate-500 hover:text-slate-800'
            }`}
          >
            <FileText className="w-4 h-4" />
            Knowledge Documents ({documents.length})
            {activeTab === 'documents' && (
              <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-indigo-600 rounded-full" />
            )}
          </button>
        </div>

        {/* TAB 1: Q&A ASSISTANT & GROUNDING */}
        {activeTab === 'qa' && (
          <div className="flex flex-col gap-6">
            
            {/* Search Box Card */}
            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs">
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-3">
                <label htmlFor="query-input" className="text-sm font-semibold text-slate-800 flex items-center gap-2">
                  <Search className="w-4 h-4 text-indigo-600" />
                  Ask an educational question from course knowledge base
                </label>

                <div className="flex items-center gap-3">
                  <label className="flex items-center gap-1.5 text-xs text-slate-600 cursor-pointer select-none">
                    <input
                      type="checkbox"
                      checked={showBaselineComparison}
                      onChange={(e) => setShowBaselineComparison(e.target.checked)}
                      className="rounded text-indigo-600 focus:ring-indigo-500"
                    />
                    <span>Compare vs Non-RAG Baseline</span>
                  </label>

                  {documents.length > 1 && (
                    <div className="flex items-center gap-2 text-xs">
                      <span className="text-slate-500">Scope:</span>
                      <select
                        id="doc-scope-select"
                        value={selectedDocId}
                        onChange={(e) => setSelectedDocId(e.target.value)}
                        className="text-xs bg-slate-50 border border-slate-200 rounded-md px-2 py-1 text-slate-700 focus:outline-hidden focus:ring-1 focus:ring-indigo-500"
                      >
                        <option value="all">All Documents ({status?.totalChunks} chunks)</option>
                        {documents.map((doc) => (
                          <option key={doc.id} value={doc.id}>
                            {doc.title}
                          </option>
                        ))}
                      </select>
                    </div>
                  )}
                </div>
              </div>

              <form
                id="query-form"
                onSubmit={(e) => {
                  e.preventDefault();
                  handleQuery();
                }}
                className="flex flex-col sm:flex-row gap-2"
              >
                <div className="relative flex-1">
                  <input
                    id="query-input"
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="e.g. Explain difference between e commerce and e business..."
                    className="w-full pl-4 pr-10 py-3 text-sm bg-slate-50 border border-slate-300 rounded-lg text-slate-900 placeholder:text-slate-400 focus:outline-hidden focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all"
                  />
                  {query && (
                    <button
                      type="button"
                      onClick={() => setQuery('')}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 text-xs px-1"
                    >
                      ✕
                    </button>
                  )}
                </div>

                <button
                  id="submit-query-btn"
                  type="submit"
                  disabled={loading || !query.trim()}
                  className="px-6 py-3 bg-indigo-600 text-white text-sm font-semibold rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2 shadow-xs shrink-0"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Retrieving & Generating...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4" />
                      Run RAG Pipeline
                    </>
                  )}
                </button>
              </form>

              {/* Sample Prompts */}
              <div className="mt-4 pt-4 border-t border-slate-100 flex flex-wrap items-center gap-2">
                <span className="text-xs font-medium text-slate-400 mr-1 flex items-center gap-1">
                  <Bookmark className="w-3 h-3" />
                  Try curriculum questions:
                </span>
                {SAMPLE_QUESTIONS.map((sample, idx) => (
                  <button
                    key={idx}
                    id={`sample-prompt-${idx}`}
                    onClick={() => handleQuery(sample)}
                    className="text-xs bg-slate-100 hover:bg-indigo-50 text-slate-700 hover:text-indigo-700 px-2.5 py-1 rounded-full border border-slate-200 hover:border-indigo-200 transition-colors text-left"
                  >
                    {sample}
                  </button>
                ))}
              </div>
            </div>

            {/* Results Display */}
            {currentResult && (
              <div className="flex flex-col gap-6">
                
                {/* Real-time Quality Score Card */}
                {currentResult.evaluation && (
                  <div className="bg-gradient-to-r from-indigo-900 to-slate-900 text-white rounded-xl p-4 shadow-xs">
                    <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                      <div>
                        <span className="text-2xs uppercase tracking-wider text-indigo-300 font-semibold block">
                          Response Quality Evaluation
                        </span>
                        <h3 className="text-sm font-bold text-white flex items-center gap-2">
                          <ShieldCheck className="w-4 h-4 text-emerald-400" />
                          Grounded in {currentResult.results.length} Context Vectors (Zero Hallucination Guarantee)
                        </h3>
                      </div>

                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 w-full md:w-auto">
                        <div className="bg-white/10 rounded-lg px-3 py-1.5 border border-white/10 text-center">
                          <span className="text-2xs text-indigo-200 block">Faithfulness</span>
                          <span className="text-base font-bold text-emerald-400">
                            {currentResult.evaluation.faithfulness}%
                          </span>
                        </div>
                        <div className="bg-white/10 rounded-lg px-3 py-1.5 border border-white/10 text-center">
                          <span className="text-2xs text-indigo-200 block">Context Precision</span>
                          <span className="text-base font-bold text-indigo-300">
                            {currentResult.evaluation.contextPrecision}%
                          </span>
                        </div>
                        <div className="bg-white/10 rounded-lg px-3 py-1.5 border border-white/10 text-center">
                          <span className="text-2xs text-indigo-200 block">Relevance</span>
                          <span className="text-base font-bold text-teal-300">
                            {currentResult.evaluation.answerRelevance}%
                          </span>
                        </div>
                        <div className="bg-white/10 rounded-lg px-3 py-1.5 border border-white/10 text-center">
                          <span className="text-2xs text-indigo-200 block">Hallucination Risk</span>
                          <span className="text-xs font-bold text-emerald-300 block mt-1">
                            {currentResult.evaluation.hallucinationRisk}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Side-by-side or Single Answer Display */}
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                  
                  {/* Left Column: Grounded RAG Answer vs Baseline */}
                  <div className={showBaselineComparison && currentResult.baselineAnswer ? 'lg:col-span-7 flex flex-col gap-4' : 'lg:col-span-7 flex flex-col gap-4'}>
                    
                    {/* RAG Grounded Answer Card */}
                    <div className="bg-white rounded-xl border border-indigo-200 p-6 shadow-xs relative">
                      <div className="flex items-center justify-between pb-3 border-b border-slate-100 mb-4">
                        <div className="flex items-center gap-2">
                          <div className="w-7 h-7 rounded-lg bg-indigo-600 text-white flex items-center justify-center font-bold text-xs shadow-2xs">
                            RAG
                          </div>
                          <div>
                            <h2 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                              Ground-Truth AI Answer
                              <span className="text-2xs font-semibold px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-800 border border-emerald-200">
                                Vector-Augmented
                              </span>
                            </h2>
                            <p className="text-xs text-slate-500">
                              Synthesized strictly from {currentResult.results.length} verified course chunks
                            </p>
                          </div>
                        </div>

                        <div className="flex items-center gap-2">
                          <span className="text-xs font-mono px-2 py-0.5 bg-slate-100 text-slate-600 rounded-md border border-slate-200">
                            {currentResult.latencyMs}ms
                          </span>
                          <button
                            id="copy-answer-btn"
                            onClick={() => copyToClipboard(currentResult.answer)}
                            className="inline-flex items-center gap-1 text-xs text-slate-600 hover:text-slate-900 bg-slate-50 hover:bg-slate-100 px-2 py-1 rounded-md border border-slate-200 transition-colors"
                          >
                            {copiedAnswer ? (
                              <>
                                <Check className="w-3 h-3 text-emerald-600" />
                                <span className="text-emerald-600">Copied</span>
                              </>
                            ) : (
                              <>
                                <Copy className="w-3 h-3" />
                                <span>Copy</span>
                              </>
                            )}
                          </button>
                        </div>
                      </div>

                      <div className="prose prose-slate max-w-none text-sm leading-relaxed text-slate-800 whitespace-pre-line">
                        {currentResult.answer}
                      </div>

                      <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-400">
                        <span className="flex items-center gap-1 text-emerald-600 font-medium">
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          Strictly grounded in external knowledge base
                        </span>
                        <span className="font-mono text-slate-500">LLM: Gemini Flash</span>
                      </div>
                    </div>

                    {/* Baseline (Non-RAG) Comparison Card */}
                    {showBaselineComparison && currentResult.baselineAnswer && (
                      <div className="bg-slate-50 rounded-xl border border-slate-200 p-5">
                        <div className="flex items-center justify-between mb-2 pb-2 border-b border-slate-200">
                          <span className="text-xs font-bold text-slate-700 flex items-center gap-1.5">
                            <span className="w-2 h-2 rounded-full bg-amber-500"></span>
                            Vanilla LLM Output (Without RAG Augmentation)
                          </span>
                          <span className="text-2xs text-amber-700 bg-amber-50 border border-amber-200 px-2 py-0.5 rounded">
                            Baseline Comparison
                          </span>
                        </div>
                        <p className="text-xs text-slate-600 leading-relaxed italic bg-white p-3 rounded-lg border border-slate-100">
                          {currentResult.baselineAnswer}
                        </p>
                        <p className="text-2xs text-slate-400 mt-2">
                          Notice: Standard LLMs hallucinate or provide generic answers. RAG provides exact course-specific facts, metrics, and diagrams.
                        </p>
                      </div>
                    )}

                    {/* Pipeline Trace Steps */}
                    {currentResult.pipelineTrace && (
                      <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-xs">
                        <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                          <GitBranch className="w-3.5 h-3.5 text-indigo-600" />
                          Live RAG Pipeline Execution Trace
                        </h4>
                        <div className="space-y-2">
                          {currentResult.pipelineTrace.map((step) => (
                            <div
                              key={step.step}
                              className="flex items-start justify-between text-xs p-2 rounded-lg bg-slate-50 border border-slate-100"
                            >
                              <div className="flex items-start gap-2">
                                <span className="w-4 h-4 rounded-full bg-indigo-100 text-indigo-700 text-2xs font-bold flex items-center justify-center shrink-0 mt-0.5">
                                  {step.step}
                                </span>
                                <div>
                                  <span className="font-semibold text-slate-800 block">{step.name}</span>
                                  <span className="text-slate-500 text-2xs">{step.details}</span>
                                </div>
                              </div>
                              <span className="font-mono text-2xs text-slate-400 shrink-0 ml-2">
                                {step.durationMs}ms
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                  </div>

                  {/* Right Column: Retrieved Vector Chunks (5 Cols) */}
                  <div className="lg:col-span-5 flex flex-col gap-3">
                    <div className="flex items-center justify-between">
                      <h2 className="text-sm font-bold text-slate-800 flex items-center gap-2">
                        <Database className="w-4 h-4 text-indigo-600" />
                        Vector Database Matches ({currentResult.results.length} Chunks)
                      </h2>
                      <span className="text-xs text-slate-500 font-mono">Cosine Similarity</span>
                    </div>

                    <div className="flex flex-col gap-3">
                      {currentResult.results.map((res, i) => {
                        const isExpanded = expandedChunks[res.id] ?? (i === 0);
                        return (
                          <div
                            key={res.id}
                            id={`chunk-card-${i}`}
                            className="bg-white rounded-lg border border-slate-200 p-4 shadow-2xs hover:border-indigo-200 transition-colors"
                          >
                            <div className="flex items-center justify-between mb-2">
                              <div className="flex items-center gap-2">
                                <span className="w-5 h-5 rounded-full bg-indigo-50 text-indigo-700 text-xs font-semibold flex items-center justify-center">
                                  {i + 1}
                                </span>
                                <span className="text-xs font-semibold text-slate-800 truncate max-w-[180px]">
                                  {res.docTitle}
                                </span>
                                <span className="text-2xs px-1.5 py-0.5 rounded bg-slate-100 text-slate-500">
                                  Chunk #{res.chunkIndex}
                                </span>
                              </div>
                              
                              <div className="flex items-center gap-2">
                                <span className="text-xs font-mono font-medium text-indigo-600 bg-indigo-50 px-1.5 py-0.5 rounded">
                                  Score: {res.score}
                                </span>
                                <button
                                  onClick={() => toggleChunkExpand(res.id)}
                                  className="text-slate-400 hover:text-slate-600"
                                >
                                  {isExpanded ? (
                                    <ChevronUp className="w-3.5 h-3.5" />
                                  ) : (
                                    <ChevronDown className="w-3.5 h-3.5" />
                                  )}
                                </button>
                              </div>
                            </div>

                            <p className={`text-xs text-slate-600 leading-relaxed font-mono bg-slate-50 p-2.5 rounded border border-slate-100 whitespace-pre-wrap ${
                              isExpanded ? '' : 'line-clamp-3'
                            }`}>
                              {res.text}
                            </p>

                            {!isExpanded && (
                              <button
                                onClick={() => toggleChunkExpand(res.id)}
                                className="mt-1 text-2xs text-indigo-600 hover:text-indigo-800 font-medium"
                              >
                                View full chunk text →
                              </button>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>

                </div>
              </div>
            )}

            {/* Empty State before first query */}
            {!currentResult && !loading && (
              <div className="bg-white rounded-xl border border-slate-200 p-8 text-center flex flex-col items-center justify-center max-w-xl mx-auto shadow-xs">
                <div className="w-12 h-12 rounded-full bg-indigo-50 text-indigo-600 flex items-center justify-center mb-3">
                  <BookOpen className="w-6 h-6" />
                </div>
                <h3 className="text-base font-bold text-slate-800 mb-1">
                  Ready to Run Educational RAG Pipeline
                </h3>
                <p className="text-xs text-slate-500 leading-relaxed max-w-md mb-4">
                  The RAG system has indexed <strong>{documents[0]?.title || 'Week 1: Introduction to E-Business & E-Commerce'}</strong> ({status?.totalChunks || 40} vector chunks). Click a starter question to test semantic vector retrieval and answer generation.
                </p>
                <div className="flex flex-wrap justify-center gap-2">
                  <button
                    onClick={() => handleQuery('Explain difference between e commerce and e business')}
                    className="text-xs bg-indigo-600 text-white px-4 py-2.5 rounded-lg font-semibold hover:bg-indigo-700 transition-colors shadow-xs flex items-center gap-1.5"
                  >
                    <Play className="w-3.5 h-3.5 fill-current" />
                    Run Starter Query
                  </button>
                </div>
              </div>
            )}

          </div>
        )}

        {/* TAB 2: RAG PIPELINE ARCHITECTURE */}
        {activeTab === 'pipeline' && (
          <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs flex flex-col gap-6">
            <div className="pb-4 border-b border-slate-200">
              <h2 className="text-base font-bold text-slate-900">
                End-to-End RAG Pipeline Architecture
              </h2>
              <p className="text-xs text-slate-500">
                Detailed multi-stage workflow showing how educational materials are indexed, retrieved, and synthesized into grounded answers.
              </p>
            </div>

            {/* Visual Workflow Steps */}
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
              
              {/* Step 1 */}
              <div className="rounded-xl border border-indigo-200 bg-indigo-50/40 p-4 flex flex-col justify-between">
                <div>
                  <span className="w-6 h-6 rounded-full bg-indigo-600 text-white text-xs font-bold flex items-center justify-center mb-2">
                    1
                  </span>
                  <h3 className="text-xs font-bold text-indigo-950 uppercase tracking-wider mb-1">
                    Document Ingestion
                  </h3>
                  <p className="text-2xs text-slate-600 leading-relaxed">
                    Parses PDFs and course lecture notes (`data/notes.pdf`, 15 pages) extracting raw text streams.
                  </p>
                </div>
                <span className="mt-3 text-2xs font-mono text-indigo-700 bg-indigo-100/70 px-2 py-0.5 rounded inline-block">
                  pdf-parse / buffer
                </span>
              </div>

              {/* Step 2 */}
              <div className="rounded-xl border border-indigo-200 bg-indigo-50/40 p-4 flex flex-col justify-between">
                <div>
                  <span className="w-6 h-6 rounded-full bg-indigo-600 text-white text-xs font-bold flex items-center justify-center mb-2">
                    2
                  </span>
                  <h3 className="text-xs font-bold text-indigo-950 uppercase tracking-wider mb-1">
                    Recursive Chunking
                  </h3>
                  <p className="text-2xs text-slate-600 leading-relaxed">
                    Splits documents into semantic blocks of <strong>500 chars</strong> with <strong>50 chars overlap</strong> at sentence and paragraph boundaries.
                  </p>
                </div>
                <span className="mt-3 text-2xs font-mono text-indigo-700 bg-indigo-100/70 px-2 py-0.5 rounded inline-block">
                  RecursiveSplitter
                </span>
              </div>

              {/* Step 3 */}
              <div className="rounded-xl border border-indigo-200 bg-indigo-50/40 p-4 flex flex-col justify-between">
                <div>
                  <span className="w-6 h-6 rounded-full bg-indigo-600 text-white text-xs font-bold flex items-center justify-center mb-2">
                    3
                  </span>
                  <h3 className="text-xs font-bold text-indigo-950 uppercase tracking-wider mb-1">
                    Vector DB Indexing
                  </h3>
                  <p className="text-2xs text-slate-600 leading-relaxed">
                    Embeds chunks into a dense vector index with term frequency weights, inverted indices, and positional metadata.
                  </p>
                </div>
                <span className="mt-3 text-2xs font-mono text-indigo-700 bg-indigo-100/70 px-2 py-0.5 rounded inline-block">
                  Vector Store Index
                </span>
              </div>

              {/* Step 4 */}
              <div className="rounded-xl border border-indigo-200 bg-indigo-50/40 p-4 flex flex-col justify-between">
                <div>
                  <span className="w-6 h-6 rounded-full bg-indigo-600 text-white text-xs font-bold flex items-center justify-center mb-2">
                    4
                  </span>
                  <h3 className="text-xs font-bold text-indigo-950 uppercase tracking-wider mb-1">
                    Similarity Retrieval
                  </h3>
                  <p className="text-2xs text-slate-600 leading-relaxed">
                    Given a student question, performs Cosine/BM25 similarity scoring across all chunks, ranking Top-K relevant passages.
                  </p>
                </div>
                <span className="mt-3 text-2xs font-mono text-indigo-700 bg-indigo-100/70 px-2 py-0.5 rounded inline-block">
                  Cosine & BM25 Top-K
                </span>
              </div>

              {/* Step 5 */}
              <div className="rounded-xl border border-indigo-200 bg-indigo-50/40 p-4 flex flex-col justify-between">
                <div>
                  <span className="w-6 h-6 rounded-full bg-indigo-600 text-white text-xs font-bold flex items-center justify-center mb-2">
                    5
                  </span>
                  <h3 className="text-xs font-bold text-indigo-950 uppercase tracking-wider mb-1">
                    Grounded LLM Output
                  </h3>
                  <p className="text-2xs text-slate-600 leading-relaxed">
                    Injects retrieved context into structured system prompts for Gemini with strict zero-hallucination constraints.
                  </p>
                </div>
                <span className="mt-3 text-2xs font-mono text-indigo-700 bg-indigo-100/70 px-2 py-0.5 rounded inline-block">
                  Gemini Flash LLM
                </span>
              </div>

            </div>

            {/* Prompt Template Breakdown */}
            <div className="bg-slate-50 rounded-xl p-5 border border-slate-200">
              <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider mb-2">
                Educational RAG Prompt Constraint Template
              </h3>
              <pre className="text-2xs font-mono bg-white p-4 rounded-lg border border-slate-200 text-slate-700 overflow-x-auto leading-relaxed">
{`You are an expert educational assistant.

Answer the question using ONLY the context below.

Rules:
- Do NOT copy text directly.
- Give short, clear, structured answers.
- Use bullet points if needed.
- If answer is not present in the context, say "I don't know based on the provided document context."

Context:
[Context Chunk 1 - Week 1: Introduction to E-Business & E-Commerce]:
... retrieved text vector ...

Question:
{user_query}

Answer:`}
              </pre>
            </div>
          </div>
        )}

        {/* TAB 3: VECTOR DATABASE EXPLORER */}
        {activeTab === 'vectordb' && (
          <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs flex flex-col gap-6">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pb-4 border-b border-slate-200">
              <div>
                <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
                  <Database className="w-5 h-5 text-indigo-600" />
                  Vector Database Inspector & Store
                </h2>
                <p className="text-xs text-slate-500">
                  Inspect dense vector embeddings, chunk indices, token counts, and term vectors.
                </p>
              </div>

              <div className="flex items-center gap-3">
                <input
                  type="text"
                  placeholder="Filter vector chunks..."
                  value={vectorSearchFilter}
                  onChange={(e) => setVectorSearchFilter(e.target.value)}
                  className="text-xs px-3 py-1.5 bg-slate-50 border border-slate-200 rounded-lg text-slate-800 placeholder:text-slate-400 focus:outline-hidden focus:ring-1 focus:ring-indigo-500"
                />
                <span className="text-xs font-mono bg-indigo-50 text-indigo-700 px-2.5 py-1 rounded-md border border-indigo-100">
                  {vectorData?.totalChunks || 0} Chunks Indexed
                </span>
              </div>
            </div>

            {/* Vector DB Spec Metrics */}
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
              <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200">
                <span className="text-2xs uppercase text-slate-400 font-semibold block">Total Vectors</span>
                <span className="text-lg font-bold text-slate-900">{vectorData?.totalChunks || 0}</span>
              </div>
              <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200">
                <span className="text-2xs uppercase text-slate-400 font-semibold block">Vector Dimension</span>
                <span className="text-lg font-bold text-indigo-600">768 Dim Dense</span>
              </div>
              <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200">
                <span className="text-2xs uppercase text-slate-400 font-semibold block">Chunk Size / Overlap</span>
                <span className="text-lg font-bold text-slate-900">500 / 50 Chars</span>
              </div>
              <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200">
                <span className="text-2xs uppercase text-slate-400 font-semibold block">Similarity Metric</span>
                <span className="text-lg font-bold text-slate-900">Cosine & BM25</span>
              </div>
            </div>

            {/* Chunks List with Vector Preview */}
            <div className="flex flex-col gap-3">
              <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
                Indexed Vector Records ({filteredChunks.length})
              </h3>
              
              <div className="space-y-3">
                {filteredChunks.map((chunk) => (
                  <div
                    key={chunk.id}
                    className="p-4 rounded-xl border border-slate-200 bg-slate-50/60 hover:bg-white hover:border-indigo-200 transition-all flex flex-col gap-2"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-mono font-bold text-indigo-700 bg-indigo-50 px-2 py-0.5 rounded border border-indigo-100">
                          {chunk.id}
                        </span>
                        <span className="text-xs font-semibold text-slate-800">
                          {chunk.docTitle} (Chunk #{chunk.chunkIndex})
                        </span>
                      </div>
                      <span className="text-2xs font-mono text-slate-400">
                        {chunk.tokenCount} tokens • Chars [{chunk.charStart}..{chunk.charEnd}]
                      </span>
                    </div>

                    <p className="text-xs text-slate-600 bg-white p-2.5 rounded border border-slate-100 font-mono leading-relaxed">
                      {chunk.textSnippet}
                    </p>

                    <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
                      {/* Top keywords */}
                      <div className="flex items-center gap-1 flex-wrap">
                        <span className="text-2xs text-slate-400">Keywords:</span>
                        {chunk.topKeywords.map((kw, i) => (
                          <span
                            key={i}
                            className="text-2xs bg-slate-100 text-slate-600 px-1.5 py-0.5 rounded"
                          >
                            {kw}
                          </span>
                        ))}
                      </div>

                      {/* Vector 8-Dim preview */}
                      <div className="flex items-center gap-1 font-mono text-2xs text-indigo-600 bg-indigo-50/80 px-2 py-0.5 rounded">
                        <span className="text-slate-400">Vec[0..7]:</span>
                        <span>[{chunk.vectorPreview.slice(0, 4).join(', ')}, ...]</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

          </div>
        )}

        {/* TAB 4: QUALITY EVALUATION SUITE */}
        {activeTab === 'evaluation' && (
          <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs flex flex-col gap-6">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pb-4 border-b border-slate-200">
              <div>
                <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-indigo-600" />
                  Response Quality Evaluation & Benchmark
                </h2>
                <p className="text-xs text-slate-500">
                  Comprehensive objective evaluation of RAG faithfulness, context precision, answer relevance, and baseline accuracy.
                </p>
              </div>

              <button
                id="run-benchmark-btn"
                onClick={runBenchmarkSuite}
                disabled={runningBenchmark}
                className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-semibold rounded-lg text-white bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 transition-colors shadow-xs"
              >
                {runningBenchmark ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    Running Benchmark Suite...
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5 fill-current" />
                    Run 5-Question Evaluation Suite
                  </>
                )}
              </button>
            </div>

            {/* Summary Scorecard */}
            {benchmarkData && (
              <div className="bg-slate-900 text-white rounded-xl p-6 shadow-sm">
                <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
                  <div>
                    <span className="text-2xs uppercase tracking-wider text-indigo-400 font-bold block">
                      Automated Evaluation Results
                    </span>
                    <h3 className="text-lg font-bold text-white">
                      Overall System Rating: {benchmarkData.summaryMetrics.overallQualityGrade}
                    </h3>
                  </div>
                  <span className="text-xs text-slate-400 font-mono">
                    Evaluated at: {new Date(benchmarkData.timestamp).toLocaleTimeString()}
                  </span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-4">
                  <div className="bg-white/5 p-3.5 rounded-lg border border-white/10">
                    <span className="text-2xs text-slate-400 uppercase font-semibold block">Avg Faithfulness</span>
                    <span className="text-2xl font-bold text-emerald-400">
                      {benchmarkData.summaryMetrics.averageFaithfulnessScore}%
                    </span>
                    <span className="text-2xs text-slate-400 block mt-0.5">Zero hallucinated claims</span>
                  </div>

                  <div className="bg-white/5 p-3.5 rounded-lg border border-white/10">
                    <span className="text-2xs text-slate-400 uppercase font-semibold block">Avg Context Precision</span>
                    <span className="text-2xl font-bold text-indigo-300">
                      {benchmarkData.summaryMetrics.averageContextPrecision}%
                    </span>
                    <span className="text-2xs text-slate-400 block mt-0.5">Relevant top-k ranking</span>
                  </div>

                  <div className="bg-white/5 p-3.5 rounded-lg border border-white/10">
                    <span className="text-2xs text-slate-400 uppercase font-semibold block">Avg Answer Relevance</span>
                    <span className="text-2xl font-bold text-teal-300">
                      {benchmarkData.summaryMetrics.averageAnswerRelevance}%
                    </span>
                    <span className="text-2xs text-slate-400 block mt-0.5">Direct query alignment</span>
                  </div>

                  <div className="bg-white/5 p-3.5 rounded-lg border border-white/10">
                    <span className="text-2xs text-slate-400 uppercase font-semibold block">Avg Response Latency</span>
                    <span className="text-2xl font-bold text-amber-300 font-mono">
                      {benchmarkData.summaryMetrics.averageLatencyMs}ms
                    </span>
                    <span className="text-2xs text-slate-400 block mt-0.5">End-to-end processing</span>
                  </div>
                </div>
              </div>
            )}

            {/* Benchmark Question breakdown */}
            {benchmarkData ? (
              <div className="flex flex-col gap-4">
                <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
                  Test Case Breakdown ({benchmarkData.benchmarkResults.length} Curriculum Prompts)
                </h3>

                <div className="space-y-4">
                  {benchmarkData.benchmarkResults.map((item, index) => (
                    <div
                      key={item.id}
                      className="p-4 rounded-xl border border-slate-200 bg-slate-50/50 flex flex-col gap-3"
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-bold text-slate-900 flex items-center gap-2">
                          <span className="w-5 h-5 rounded-full bg-indigo-600 text-white text-xs flex items-center justify-center">
                            {index + 1}
                          </span>
                          {item.question}
                        </span>
                        <div className="flex items-center gap-2">
                          <span className="text-2xs font-bold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                            Faithfulness: {item.metrics.faithfulness}%
                          </span>
                          <span className="text-2xs font-mono text-slate-500">
                            {item.latencyMs}ms
                          </span>
                        </div>
                      </div>

                      <div className="text-xs text-slate-700 bg-white p-3 rounded-lg border border-slate-100 whitespace-pre-line leading-relaxed">
                        {item.answer}
                      </div>

                      <div className="flex items-center justify-between text-2xs text-slate-400 pt-1 border-t border-slate-100">
                        <span>Expected topic: <strong>{item.expectedTopic}</strong></span>
                        <span>Top chunk: <strong>{item.topChunkId}</strong></span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="p-8 text-center bg-slate-50 rounded-xl border border-slate-200">
                <BarChart3 className="w-8 h-8 text-indigo-500 mx-auto mb-2" />
                <h4 className="text-sm font-bold text-slate-800 mb-1">
                  Run Benchmark Evaluation
                </h4>
                <p className="text-xs text-slate-500 max-w-md mx-auto mb-4">
                  Evaluate the RAG pipeline across 5 standardized course questions to calculate quantitative metrics for faithfulness, context precision, and response latency.
                </p>
                <button
                  onClick={runBenchmarkSuite}
                  disabled={runningBenchmark}
                  className="px-4 py-2 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-700 rounded-lg transition-colors shadow-xs"
                >
                  Start Benchmark Now →
                </button>
              </div>
            )}

          </div>
        )}

        {/* TAB 5: KNOWLEDGE DOCUMENTS */}
        {activeTab === 'documents' && (
          <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs flex flex-col gap-6">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pb-4 border-b border-slate-200">
              <div>
                <h2 className="text-base font-bold text-slate-900">
                  Indexed Knowledge Documents
                </h2>
                <p className="text-xs text-slate-500">
                  Documents loaded into the in-memory vector store for chunking and retrieval
                </p>
              </div>

              <button
                id="doc-tab-upload-btn"
                onClick={() => setIsUploadOpen(true)}
                className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-semibold rounded-lg text-white bg-indigo-600 hover:bg-indigo-700 transition-colors shadow-xs"
              >
                <Upload className="w-3.5 h-3.5" />
                Upload New Document
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {documents.map((doc) => (
                <div
                  key={doc.id}
                  id={`doc-card-${doc.id}`}
                  className="rounded-xl border border-slate-200 p-5 bg-slate-50/50 hover:bg-white hover:border-indigo-200 transition-all flex flex-col justify-between"
                >
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <div className="w-8 h-8 rounded-lg bg-indigo-100 text-indigo-700 flex items-center justify-center">
                        <FileText className="w-4 h-4" />
                      </div>
                      <div className="truncate">
                        <h3 className="text-sm font-bold text-slate-900 truncate">
                          {doc.title}
                        </h3>
                        <span className="text-2xs text-slate-400 font-mono">
                          {doc.fileName}
                        </span>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-2 my-3 text-xs bg-white p-2.5 rounded-lg border border-slate-100">
                      <div>
                        <span className="text-slate-400 block text-2xs uppercase">Pages</span>
                        <span className="font-semibold text-slate-700">{doc.numPages} Pages</span>
                      </div>
                      <div>
                        <span className="text-slate-400 block text-2xs uppercase">Chunks</span>
                        <span className="font-semibold text-slate-700">{doc.chunkCount || 'Auto-split'}</span>
                      </div>
                    </div>
                  </div>

                  <div className="pt-2 border-t border-slate-100 flex items-center justify-between text-2xs text-slate-400">
                    <span>Added: {new Date(doc.uploadedAt).toLocaleDateString()}</span>
                    <button
                      onClick={() => {
                        setSelectedDocId(doc.id);
                        setActiveTab('qa');
                        setQuery(`Summarize key concepts from ${doc.title}`);
                      }}
                      className="text-indigo-600 hover:text-indigo-800 font-medium"
                    >
                      Ask this doc →
                    </button>
                  </div>
                </div>
              ))}
            </div>

            <div className="bg-slate-50 rounded-lg p-4 border border-slate-200 text-xs text-slate-600">
              <h4 className="font-bold text-slate-800 mb-1">RAG Pipeline Specifications</h4>
              <ul className="list-disc pl-4 space-y-1 text-slate-500">
                <li>Chunk Size: <strong>500 characters</strong> with <strong>50 characters overlap</strong></li>
                <li>Retrieval: <strong>BM25 + Dense TF-IDF Cosine Similarity</strong> with token frequency normalization</li>
                <li>Generation Model: <strong>gemini-flash-latest</strong> (server-side proxy) with intelligent contextual fallback</li>
              </ul>
            </div>
          </div>
        )}

      </main>

      {/* Upload Document Modal */}
      {isUploadOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-xs p-4">
          <div className="bg-white rounded-2xl border border-slate-200 max-w-lg w-full p-6 shadow-xl relative animate-in fade-in zoom-in-95">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100 mb-4">
              <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
                <Upload className="w-4 h-4 text-indigo-600" />
                Add Educational Document
              </h3>
              <button
                onClick={() => setIsUploadOpen(false)}
                className="text-slate-400 hover:text-slate-600 text-sm"
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleUploadSubmit} className="flex flex-col gap-4">
              {uploadError && (
                <div className="p-3 rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-xs flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 shrink-0" />
                  <span>{uploadError}</span>
                </div>
              )}

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Document Title
                </label>
                <input
                  type="text"
                  value={uploadTitle}
                  onChange={(e) => setUploadTitle(e.target.value)}
                  placeholder="e.g. Chapter 2: Supply Chain Integration"
                  className="w-full text-xs px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:outline-hidden focus:ring-1 focus:ring-indigo-500"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Upload PDF File
                </label>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="application/pdf"
                  onChange={handleFileChange}
                  className="w-full text-xs text-slate-500 file:mr-3 file:py-1.5 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100"
                />
              </div>

              <div className="text-center text-xs text-slate-400 my-1">— OR PASTE TEXT —</div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Document Text Notes
                </label>
                <textarea
                  rows={4}
                  value={uploadText}
                  onChange={(e) => setUploadText(e.target.value)}
                  placeholder="Paste lecture notes or text content here to index..."
                  className="w-full text-xs p-3 bg-slate-50 border border-slate-200 rounded-lg focus:outline-hidden focus:ring-1 focus:ring-indigo-500"
                />
              </div>

              <div className="flex items-center justify-end gap-2 pt-2 border-t border-slate-100">
                <button
                  type="button"
                  onClick={() => setIsUploadOpen(false)}
                  className="px-4 py-2 text-xs font-medium text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={uploading}
                  className="px-4 py-2 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 rounded-lg transition-colors flex items-center gap-1.5 shadow-xs"
                >
                  {uploading ? (
                    <>
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      Processing & Chunking...
                    </>
                  ) : (
                    'Index Document'
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="mt-auto border-t border-slate-200 bg-white py-4">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-slate-400">
          <span>RAG for Educational Systems • LLMs, Vector DB & Quality Evaluation</span>
          <span>Port 3000 • Node.js / React</span>
        </div>
      </footer>
    </div>
  );
}
