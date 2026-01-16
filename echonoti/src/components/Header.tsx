import Link from "next/link";
import { Rss, Mic } from "lucide-react";

export function Header() {
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
              Notifications
            </Link>
            <Link
              href="/bookmarked"
              className="transition-colors hover:text-foreground/80 text-foreground/60"
            >
              Bookmarked
            </Link>
            <Link
              href="/voice"
              className="transition-colors hover:text-foreground/80 text-foreground/60 flex items-center gap-1"
            >
              <Mic className="h-4 w-4" />
              Voice
            </Link>
          </nav>
        </div>
      </div>
    </header>
  );
}
