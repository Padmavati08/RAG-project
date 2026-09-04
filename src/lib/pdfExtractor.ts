// Client-side PDF text extraction utility
import * as pdfjsLib from 'pdfjs-dist';

// Configure pdfjs worker to use unpkg or cdnjs, or disable worker for client-side synchronous parsing
if (typeof window !== 'undefined') {
  // Use official cdnjs worker matching installed version or fallback
  pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version || '4.10.38'}/pdf.worker.min.mjs`;
}

export async function extractTextFromPdfFile(file: File): Promise<{ text: string; numPages: number }> {
  try {
    const arrayBuffer = await file.arrayBuffer();
    const loadingTask = pdfjsLib.getDocument({
      data: new Uint8Array(arrayBuffer),
      useSystemFonts: true,
    } as any);

    const pdfDoc = await loadingTask.promise;
    const numPages = pdfDoc.numPages;
    const pageTextPromises: Promise<string>[] = [];

    for (let i = 1; i <= numPages; i++) {
      pageTextPromises.push(
        (async () => {
          try {
            const page = await pdfDoc.getPage(i);
            const content = await page.getTextContent();
            return content.items
              .map((item: any) => item.str || '')
              .join(' ');
          } catch {
            return '';
          }
        })()
      );
    }

    const pages = await Promise.all(pageTextPromises);
    const fullText = pages.filter(Boolean).join('\n\n');
    return {
      text: fullText.trim(),
      numPages,
    };
  } catch (err) {
    console.warn('[pdfExtractor] Fallback to binary text scanning:', err);
    // Fallback: extract any visible ASCII text from the PDF buffer directly
    const text = await extractTextFromPdfFallback(file);
    return {
      text,
      numPages: 1,
    };
  }
}

async function extractTextFromPdfFallback(file: File): Promise<string> {
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let asciiText = '';
  let insideParenthesis = false;
  let currentWord = '';

  for (let i = 0; i < bytes.length; i++) {
    const char = String.fromCharCode(bytes[i]);
    if (char === '(' && !insideParenthesis) {
      insideParenthesis = true;
      currentWord = '';
    } else if (char === ')' && insideParenthesis) {
      insideParenthesis = false;
      if (currentWord.length > 2) {
        asciiText += currentWord + ' ';
      }
      currentWord = '';
    } else if (insideParenthesis) {
      if (bytes[i] >= 32 && bytes[i] <= 126) {
        currentWord += char;
      }
    }
  }

  const cleaned = asciiText
    .replace(/[^\x20-\x7E\n]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  return cleaned.length > 50 ? cleaned : '';
}
