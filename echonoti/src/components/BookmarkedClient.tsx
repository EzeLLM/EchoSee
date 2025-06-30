"use client";

import { Bookmark } from "lucide-react";
import { NotificationList } from "./NotificationList";
import { useT } from "@/hooks/use-i18n";
import type { Notification } from "@/lib/types";

export function BookmarkedClient({ notifications }: { notifications: Notification[] }) {
  const t = useT();
  return (
    <div className="w-full">
      <h1 className="text-3xl font-bold mb-6 font-headline tracking-tighter">
        {t("bookmarked")}
      </h1>
      {notifications.length > 0 ? (
        <NotificationList initialNotifications={notifications} />
      ) : (
        <div className="text-center py-12 text-muted-foreground border-2 border-dashed rounded-lg h-full flex flex-col justify-center items-center">
          <Bookmark className="mx-auto h-12 w-12 text-muted-foreground/50" />
          <h3 className="mt-4 text-lg font-semibold">{t("no_bookmarks")}</h3>
          <p className="mt-1 text-sm">{t("click_to_bookmark")}</p>
        </div>
      )}
    </div>
  );
}
