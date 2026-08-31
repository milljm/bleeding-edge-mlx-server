import type { ComponentProps } from "react";
import * as LabelPrimitive from "@radix-ui/react-label";
import { cn } from "@/lib/utils";

function Label({ className, ...props }: ComponentProps<typeof LabelPrimitive.Root>) {
  return (
    <LabelPrimitive.Root
      className={cn("text-xs font-medium tracking-wide text-muted-foreground", className)}
      {...props}
    />
  );
}

export { Label };
