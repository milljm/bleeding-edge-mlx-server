import { Fragment, useState, type ReactNode } from "react";
import { Check, Copy } from "lucide-react";
import { highlightCode, normalizeLang } from "@/lib/chat/highlight";
import { parseTableAt, type MdTable, type TableAlign } from "@/lib/chat/md-table";
import { PYGMENTS_STYLES, SYNTAX_AUTO, useSyntaxPref } from "@/lib/chat/syntax";
import { cn } from "@/lib/utils";

function parseFenceInfo(info: string): { lang: string; file: string | null } {
  const raw = info.trim();
  if (!raw) return { lang: "text", file: null };
  const named = raw.match(/(?:filename|file|title|path)\s*[:=]\s*["']?([^\s"']+)/i);
  const colon = raw.match(/^([A-Za-z0-9_+-]+)\s*:\s*(\S+)$/);
  if (colon) return { lang: colon[1] ?? "text", file: colon[2] ?? null };
  const parts = raw.split(/\s+/).filter(Boolean);
  const first = parts[0] ?? "text";
  if (named?.[1]) return { lang: first.includes("/") ? "text" : first, file: named[1] };
  if (first.includes("/") || /\.\w+$/.test(first)) {
    const ext = first.split(".").pop() ?? "text";
    return { lang: ext, file: first };
  }
  return { lang: first, file: null };
}

function safeHref(raw: string): string | null {
  const url = raw.trim();
  if (!url) return null;
  if (/^(https?:\/\/|mailto:|\/|#)/i.test(url)) return url;
  return null;
}

function safeImg(raw: string): string | null {
  const url = raw.trim();
  if (!url) return null;
  if (/^https?:\/\//i.test(url) || url.startsWith("data:image/")) return url;
  return null;
}

function inline(text: string): ReactNode[] {
  const parts: ReactNode[] = [];
  const re =
    /(`[^`]+`)|(!\[([^\]]*)\]\(\s*<?([^)\s>]+)>?(?:\s+"([^"]*)")?\s*\))|(\[([^\]]+)\]\(\s*<?([^)\s>]+)>?(?:\s+"([^"]*)")?\s*\))|(\*\*[^*]+\*\*)|(\*[^*]+\*)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let k = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    if (m[1]) {
      parts.push(
        <code key={k++} className="rounded-xs bg-code px-1 py-0.5 font-mono text-xs text-code-fg">
          {m[1].slice(1, -1)}
        </code>,
      );
    } else if (m[2]) {
      const src = safeImg(m[4] ?? "");
      const alt = m[3] ?? "";
      if (src) {
        parts.push(
          <img
            key={k++}
            src={src}
            alt={alt}
            title={m[5] || alt}
            className="my-2 max-h-64 max-w-full rounded-md outline outline-1 -outline-offset-1 outline-foreground/10"
          />,
        );
      } else {
        parts.push(m[2]);
      }
    } else if (m[6]) {
      const href = safeHref(m[8] ?? "");
      const label = m[7] ?? "";
      if (href) {
        parts.push(
          <a
            key={k++}
            href={href}
            title={m[9] || undefined}
            target="_blank"
            rel="noreferrer"
            className="underline underline-offset-2 hover:text-foreground"
          >
            {label}
          </a>,
        );
      } else {
        parts.push(m[6]);
      }
    } else if (m[10]) {
      parts.push(
        <strong key={k++} className="font-medium text-foreground">
          {m[10].slice(2, -2)}
        </strong>,
      );
    } else {
      parts.push(
        <em key={k++} className="italic">
          {(m[11] ?? "").slice(1, -1)}
        </em>,
      );
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

const HEADING_CLASS = [
  "mt-4 mb-2 text-xl font-semibold tracking-tight text-foreground first:mt-0",
  "mt-4 mb-2 text-lg font-semibold text-foreground first:mt-0",
  "mt-3 mb-1.5 text-base font-semibold text-foreground",
  "mt-3 mb-1 text-sm font-semibold text-foreground",
  "mt-2 mb-1 text-sm font-medium text-foreground",
  "mt-2 mb-1 text-sm font-medium text-muted-foreground",
];

function renderProse(block: string, nodes: ReactNode[]) {
  const lines = block.split("\n");
  let para: string[] = [];
  let list: string[] = [];
  let quote: string[] = [];
  const flushPara = () => {
    if (!para.length) return;
    nodes.push(
      <p key={`p${nodes.length}`} className="my-2 whitespace-pre-wrap leading-relaxed">
        {inline(para.join("\n"))}
      </p>,
    );
    para = [];
  };
  const flushList = () => {
    if (!list.length) return;
    nodes.push(
      <ul key={`l${nodes.length}`} className="my-2 list-disc space-y-1 pl-5">
        {list.map((item, idx) => (
          <li key={idx}>{inline(item)}</li>
        ))}
      </ul>,
    );
    list = [];
  };
  const flushQuote = () => {
    if (!quote.length) return;
    nodes.push(
      <blockquote key={`q${nodes.length}`} className="my-2 border-l-2 border-border pl-3 text-muted-foreground">
        {inline(quote.join("\n"))}
      </blockquote>,
    );
    quote = [];
  };

  for (let i = 0; i < lines.length; i++) {
    const table = parseTableAt(lines, i);
    if (table) {
      flushPara();
      flushList();
      flushQuote();
      nodes.push(<MarkdownTable key={`t${nodes.length}`} table={table.table} />);
      i += table.consumed - 1;
      continue;
    }
    const line = lines[i] ?? "";
    const heading = line.match(/^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$/);
    if (heading) {
      flushPara();
      flushList();
      flushQuote();
      const level = Math.min(6, heading[1].length);
      const Tag = (`h${level}` as unknown) as "h1";
      nodes.push(
        <Tag key={`h${nodes.length}`} className={HEADING_CLASS[level - 1]}>
          {inline(heading[2] ?? "")}
        </Tag>,
      );
      continue;
    }
    if (/^ {0,3}(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
      flushPara();
      flushList();
      flushQuote();
      nodes.push(<hr key={`r${nodes.length}`} className="my-4 border-border" />);
      continue;
    }
    const quoted = line.match(/^ {0,3}>\s?(.*)$/);
    if (quoted) {
      flushPara();
      flushList();
      quote.push(quoted[1] ?? "");
      continue;
    }
    const bullet = line.match(/^\s*[-*]\s+(.*)$/);
    if (bullet) {
      flushPara();
      flushQuote();
      list.push(bullet[1] ?? "");
      continue;
    }
    if (line.trim() === "") {
      flushList();
      flushPara();
      flushQuote();
      continue;
    }
    flushList();
    flushQuote();
    para.push(line);
  }
  flushList();
  flushQuote();
  flushPara();
}

function alignClass(align: TableAlign | undefined): string {
  if (align === "center") return "text-center";
  if (align === "right") return "text-right";
  return "text-left";
}

function MarkdownTable({ table }: { table: MdTable }) {
  return (
    <div className="my-3 overflow-x-auto rounded-md outline outline-1 -outline-offset-1 outline-foreground/10">
      <table className="w-full min-w-max border-collapse text-sm">
        <thead>
          <tr className="border-b border-border bg-secondary/70">
            {table.headers.map((h, i) => (
              <th
                key={i}
                className={cn(
                  "px-3 py-1.5 text-xs font-medium tracking-wide text-foreground",
                  alignClass(table.align[i]),
                )}
              >
                {inline(h)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row, r) => (
            <tr key={r} className="border-b border-border/50 last:border-0">
              {row.map((cell, c) => (
                <td
                  key={c}
                  className={cn("px-3 py-1.5 text-foreground/85", alignClass(table.align[c]))}
                >
                  {inline(cell)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CodeBlock({ lang, body, file }: { lang: string; body: string; file: string | null }) {
  const name = normalizeLang(lang);
  const [copied, setCopied] = useState(false);
  const [syntax, setSyntax] = useSyntaxPref();
  const label = file || (name && name !== "text" ? name : "code");

  async function copy() {
    try {
      await navigator.clipboard.writeText(body);
    } catch {
      /* ignore */
    }
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1400);
  }

  return (
    <div className="my-3 overflow-hidden rounded-md bg-code text-code-fg outline outline-1 -outline-offset-1 outline-foreground/10">
      <div className="flex items-center justify-between gap-2 border-b border-border px-2 py-0.5">
        <span className="min-w-0 truncate px-1 font-mono text-[10px] uppercase tracking-wide text-muted-foreground">
          {label}
        </span>
        <div className="flex items-center">
          <select
            aria-label="Syntax highlighting theme"
            value={syntax}
            onChange={(e) => setSyntax(e.target.value)}
            className="mr-0.5 h-8 max-w-[8.5rem] cursor-pointer truncate rounded-sm bg-transparent px-1 font-mono text-[10px] text-muted-foreground outline-none hover:text-foreground"
          >
            <option value={SYNTAX_AUTO}>auto</option>
            {PYGMENTS_STYLES.map((s) => (
              <option key={s.id} value={s.id}>
                {s.id}
              </option>
            ))}
          </select>
          <button
            type="button"
            aria-label={copied ? "Copied" : "Copy code"}
            className="inline-flex size-8 items-center justify-center rounded-sm text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
            onClick={() => void copy()}
          >
            {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
          </button>
        </div>
      </div>
      <pre className="overflow-x-auto p-3 font-mono text-xs leading-relaxed">
        <code dangerouslySetInnerHTML={{ __html: highlightCode(body, name) }} />
      </pre>
    </div>
  );
}

function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

const OPEN_FENCE = /^( {0,3})(`{3,}|~{3,})([^\n]*)$/;

export function Markdown({ text, className }: { text: string; className?: string }) {
  const source = typeof text === "string" ? text : String(text ?? "");
  const lines = source.split("\n");
  const nodes: ReactNode[] = [];
  let prose: string[] = [];
  let i = 0;

  const flushProse = () => {
    if (!prose.length) return;
    renderProse(prose.join("\n"), nodes);
    prose = [];
  };

  while (i < lines.length) {
    const open = lines[i].match(OPEN_FENCE);
    if (open) {
      flushProse();
      const marker = open[2];
      const meta = parseFenceInfo(open[3] ?? "");
      const close = new RegExp(`^ {0,3}${escapeRe(marker)}[ \\t]*$`);
      const body: string[] = [];
      i += 1;
      while (i < lines.length && !close.test(lines[i])) {
        body.push(lines[i]);
        i += 1;
      }
      if (i < lines.length) i += 1;
      nodes.push(
        <CodeBlock key={`c${nodes.length}`} lang={meta.lang} file={meta.file} body={body.join("\n")} />,
      );
      continue;
    }
    prose.push(lines[i]);
    i += 1;
  }
  flushProse();

  return (
    <div className={cn("text-pretty text-sm text-foreground/90", className)}>
      {nodes.length ? nodes : <Fragment>{inline(source)}</Fragment>}
    </div>
  );
}
