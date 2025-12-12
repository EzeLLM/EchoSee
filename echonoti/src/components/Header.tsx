"use client";

import Link from "next/link";
import { Rss } from "lucide-react";
import { useT } from "@/hooks/use-i18n";

export function Header() {
  const t = useT();
  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-14 max-w-screen-2xl items-center">
        <div className="mr-4 flex items-center">
          <Link href="/" className="mr-6 flex items-center space-x-2">
            <Rss className="h-6 w-6 text-primary" />
            <span className="font-bold sm:inline-block font-headline">
              EchoNoti
            </span>
          </Link>
          <nav className="flex items-center gap-6 text-sm">
            <Link
              href="/"
              className="transition-colors hover:text-foreground/80 text-foreground/60"
            >
              {t("notifications")}
            </Link>
            <Link
              href="/bookmarked"
              className="transition-colors hover:text-foreground/80 text-foreground/60"
            >
              {t("bookmarked")}
            </Link>
            <Link
              href="/settings"
              className="transition-colors hover:text-foreground/80 text-foreground/60"
            >
              {t("settings")}
            </Link>
            <Link
              href="/about"
              className="transition-colors hover:text-foreground/80 text-foreground/60"
            >
              {t("about")}
            </Link>
          </nav>
        </div>
      </div>
    </header>
  );
}
