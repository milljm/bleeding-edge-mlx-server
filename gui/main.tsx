import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { AppShell } from "@/components/studio/app-shell";
import { applyTheme, readThemePref } from "@/lib/theme";
import "@/styles.css";

applyTheme(readThemePref());

const root = document.getElementById("root");
if (!root) throw new Error("Edge GUI root missing");
createRoot(root).render(
  <StrictMode>
    <AppShell />
  </StrictMode>,
);
