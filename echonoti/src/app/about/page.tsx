import fs from 'fs/promises';
import path from 'path';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export const dynamic = 'force-dynamic';

export default async function AboutPage() {
  const readmePath = path.join(process.cwd(), '..', 'README.md');
  const markdown = await fs.readFile(readmePath, 'utf-8');

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold mb-4 font-headline">About EchoSee</h1>
      <div className="prose dark:prose-invert max-w-none">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{markdown}</ReactMarkdown>
        <p className="mt-6">
          Repo:{' '}
          <a
            href="https://github.com/EzeLLM/EchoSee"
            className="text-primary underline"
          >
            https://github.com/EzeLLM/EchoSee
          </a>
        </p>
        <p>
          Support:{' '}
          <a href="mailto:ezel964@icloud.com" className="text-primary underline">
            ezel964@icloud.com
          </a>
        </p>
      </div>
    </div>
  );
}
