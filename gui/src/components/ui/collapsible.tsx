import { useState, type ReactNode } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * Collapsible bubble card used to group settings. The title row toggles the
 * body; an open bubble tints its title with the theme's open accent
 * (orange in dark mode, green in light mode — see --edge-open in styles.css).
 */
export function Collapsible({
  title,
  hint,
  defaultOpen = false,
  footer,
  children,
}: {
  title: string;
  hint?: ReactNode;
  defaultOpen?: boolean;
  footer?: ReactNode;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="rounded-2xl bg-card px-5 py-4 shadow-[var(--shadow-border)]">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-3 text-left"
      >
        <span className="min-w-0">
          <span
            className={cn(
              "block text-xs font-medium uppercase tracking-wide transition-colors",
              open ? "text-open" : "text-muted-foreground",
            )}
          >
            {title}
          </span>
          {hint ? <span className="mt-1 block max-w-lg text-sm text-muted-foreground">{hint}</span> : null}
        </span>
        <ChevronDown
          className={cn(
            "size-4 shrink-0 text-muted-foreground transition-transform duration-200",
            open && "rotate-180",
          )}
        />
      </button>
      <div
        className={cn(
          "grid transition-[grid-template-rows] duration-200 ease-[var(--ease-smooth-out)]",
          open ? "grid-rows-[1fr]" : "grid-rows-[0fr]",
        )}
      >
        <div className={cn("min-h-0 overflow-hidden", open && "pt-4")}>
          <div className="space-y-4">{children}</div>
          {footer}
        </div>
      </div>
    </section>
  );
}